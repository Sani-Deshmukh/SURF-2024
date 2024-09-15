##This code takes M3 data and uses feature matching to register it to LOLA/Kaguya topography within 60 degrees of the equator, and LOLA data poleward of 60 degrees.
##The first half of the code prepares the Radiance and Topography for matching
##The second half of the code matches an averaged Radiance file with a hillshade map generated with similar illumination conditions.
##If the first match does not work, the code iteratively creates hillshades with different lighting until a match is successful.

##The code uses M3 data that have been clipped to 2677 rows to be more similar to HVM3 data. The data are all in Data_M3/. The middle 2677 rows were extracted for each M3 product.

#To execute the code, run the script with the M3 image ID as the one argument, i.e. "./LTB_FeatureMatching_M3.sh M3G20090207T090807" (no quotes)

##Jay Dickson (jdickson@umn.edu)
##August, 2023

#The code expects one argument: the M3 image ID.


import os, sys, shutil
from PIL import Image
from osgeo import gdal, osr, gdalconst
from osgeo_utils import gdal_calc
import numpy as np
import pandas as pd
from FM_toolkit import *

DF_COLS = ['M3ID', 'WORKED', 
           'FIRST_TRY', 'FILE_FOUND',
           'LON', 'LON_MIN', 'LON_MAX',  
           'LAT', 'LAT_MIN', 'LAT_MAX',
           'P_X', 'P_X_MIN', 'P_X_MAX',
           'P_Y', 'P_Y_MIN', 'P_Y_MAX',
           'REPR_LON', 'REPR_LON_MIN', 'REPR_LON_MAX',  
           'REPR_LAT', 'REPR_LAT_MIN', 'REPR_LAT_MAX',
           'INC', 'AZM', 'INC_MATCH', 'AZM_MATCH']

def run_match(m3id):
    '''
        Runs feature matching for a given m3id
    '''
    infodict = {}
    for k in DF_COLS: infodict[k] = None
    infodict["M3ID"]=m3id

    #Check to see if this image has already been matched or attempted to match. 
    # TODO: EXPLAIN WHY THIS DOES WHAT IT DOES
    work = any(x.startswith(m3id) for x in os.listdir('Results/Worked'))
    fail = any(x.startswith(m3id) for x in os.listdir('Results/Failed'))
    if work:
        shutil.rmtree(f'Results/Worked/{m3id}')
    elif fail:
        if f'{m3id}.txt' in os.listdir('Results/Failed'):
            os.remove(f'Results/Failed/{m3id}.txt')
        else:
            os.remove(f'Results/Failed/{m3id}_RDN_average_byte.tif')
    
    #Input topography file to be used to generate hillshade
    inlola = "Topography/LunarTopography_60mpx.tif"

    #Ground resolution of M3 in m/px.
    #This is required for extraction of LOLA data over a reasonable area of the Moon to generate a hillshade for matching.
    m3resolution = 200

    #Define starting and ending bands in M3 for averaging.
    #The code averages M3 bands together to increase the quality of the matching product. Set the start band and end band (in units of band number)
    band_bnds=(71,81)

    #Input directory for radiance data. This is a volume of M3 data that have been clipped to 2677 rows to be more similar to HVM3 data. The middle 2677 rows were extracted for each M3 product.
    m3dir="Data_M3"

    #Create a directory to run all of the processes/make files in.
    workdir=f'{m3id}_work'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    #Source files
    inrdn=f"{m3dir}/{m3id}_RDN.IMG"
    inobs=f"{m3dir}/{m3id}_OBS.IMG"
    inloc=f"{m3dir}/{m3id}_LOC.IMG"

    loc_img = gdal.Open(inloc)
    rdn_img = gdal.Open(inrdn)
    obs_img = gdal.Open(inobs)

    infodict['FILE_FOUND']=True
    # Check to make sure the rdn loc and obs images are all found
    if loc_img is None or rdn_img is None or obs_img is None:
        print(f"M3ID NOT FOUND: {m3id}")
        if os.path.isdir(workdir):
            shutil.rmtree(workdir)
        f = open(f"Results/Failed/{m3id}.txt", "a")
        f.write(f"{m3id} NOT FOUND")
        f.close()

        infodict['FILE_FOUND'] = False
        infodict['WORKED'] = False
        return infodict

    ###Calculate extent for hillshade. This calculates the center lat/lon from the LOC file for M3.
    ###For HVM3, the LOC file will not be available at this stage, so center coordinates should be extracted from the filename, or from a lookup table of targets.

    #Calculate center latitude/longitude
    latband = loc_img.GetRasterBand(2)
    minlat, maxlat = latband.ComputeStatistics(0)[:2]
    clat = (minlat + maxlat)/2

    lonband = loc_img.GetRasterBand(1)
    minlon, maxlon =  lonband.ComputeStatistics(0)[:2]
    clon = (minlon + maxlon)/2

    infodict['LAT'] = clat
    infodict['LAT_MIN'] = minlat
    infodict['LAT_MAX'] = maxlat
    infodict['LON'] = clon
    infodict['LON_MIN'] = minlon
    infodict['LON_MAX'] = maxlon

    #Get x and y for clon and clat in meter space 
    projin = osr.SpatialReference()
    projout = osr.SpatialReference()
    projin.SetFromUserInput('+proj=longlat +a=1737400 +b=1737400 +no_defs')
    projout.SetFromUserInput(f'+proj=sinu +lon_0={clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs')

    in2out_tf = osr.CoordinateTransformation(projin, projout)
    coords = in2out_tf.TransformPoint(clon, clat, 0)
    x, y = coords[:2]

    #Determine the number of lines and samples in the image.
    #These values are multiplied by the resolution to create the bounding box to use to clip LOLA to the approximate area of the image.
    samples, lines = rdn_img.RasterXSize, rdn_img.RasterYSize
    
    #Calculate the width and height of the image in meters using M3's resolution (defined at the beginning of the script)
    fullwidth = samples * m3resolution
    fullheight = lines * m3resolution

    #Set the bounding box. Add a kilometer on all sides for margin (may need more if pointing uncertainty is very high)
    xmin, xmax = x-fullwidth/2-1000, x+fullwidth/2+1000
    ymin, ymax = y-fullheight/2-1000, y+fullheight/2+1000

    infodict['P_X'] = x 
    infodict['P_X_MIN'] = xmin
    infodict['P_X_MAX'] = xmax
    infodict['P_Y'] = y
    infodict['P_Y_MIN'] = ymin 
    infodict['P_Y_MAX'] = ymax

    #Get x and y in degree space. Parse the lines to define boudning box coordinates.
    out2in_tf = osr.CoordinateTransformation(projout, projin)
    minlon, maxlat = out2in_tf.TransformPoint(xmin, ymax, 0)[:2]
    maxlon = out2in_tf.TransformPoint(xmax, ymax, 0)[0]
    minlat = out2in_tf.TransformPoint(xmin, ymin, 0)[1]

    infodict['REPR_LAT'] = (minlat+maxlat)/2
    infodict['REPR_LAT_MIN'] = minlat
    infodict['REPR_LAT_MAX'] = maxlat
    infodict['REPR_LON'] = (minlon+maxlon)/2
    infodict['REPR_LON_MIN'] = minlon
    infodict['REPR_LON_MAX'] = maxlon

    #Clip the global topography to the bounding box. The LOLA DEM is 60 m/px and does not need to be resampled for this procedure. Only clip.
    gdal.Warp(f'{workdir}/{m3id}_topo.tif', inlola,
            options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(minlon,minlat,maxlon,maxlat),
                outputBoundsSRS='+ =longlat +a=1737400 +b=1737400 +no_defs'
            ))
	
    #Calculate the mean solar azimuth and solar incidence from the Observation data file
	#This will not be available for HVM3. Will need predicted values for center of the planned target from MOS/GDS or MdNav
    azmband = obs_img.GetRasterBand(1)
    azmmean = azmband.ComputeStatistics(0)[2]
    incband = obs_img.GetRasterBand(2)
    incmean = incband.ComputeStatistics(0)[2]

    infodict['INC'] = incmean
    infodict['AZM'] = azmmean

	#Reproject topography to sinusoidal to increase chances of a match. Can't get the $clon variable in the proj4 syntax, so sending it to temp file, then executing that.
    gdal.Warp(f'{workdir}/{m3id}_topo_sinu.tif',f'{workdir}/{m3id}_topo.tif',
            options=gdal.WarpOptions(
                format='GTiff',
                dstSRS=f'+proj=sinu +lon_0={clon} +x_0=0 +y_0=0 +a=1737400 +b=1737400 +no_defs'
            ))
    
    #Create a flat Radiance product with 10 bands averaged. For M3 testing, longer wavelengths produced better matching results.
	#Separate the individual bands into separate rasters
    bfns = [f'{workdir}/{m3id}_RDN_b{b}.tif' for b in range(*band_bnds)]
    for b in range(*band_bnds):
        gdal.Translate(bfns[b-band_bnds[0]], inrdn,
                    options=gdal.TranslateOptions(
                        format='GTiff',
                        bandList=[b]
                    ))
    gdal_calc.Calc(calc='mean(a, axis=0)', a=bfns, outfile=f'{workdir}/{m3id}_RDN_average.tif')

    #Create 8-bit raster of averaged radiance for matching
    med_im = np.array(Image.open(f'{workdir}/{m3id}_RDN_average.tif'))
    gdal.Translate(f"{workdir}/{m3id}_RDN_average_byte.tif", f"{workdir}/{m3id}_RDN_average.tif", 
                options=gdal.TranslateOptions(
                    format='GTiff',
                    outputType=gdalconst.GDT_Byte,
                    scaleParams=[[med_im.min(),med_im.max(),1,255]]
                ))

    # Clean up unused files
    os.remove(f"{workdir}/{m3id}_RDN_average.tif")
    for name in bfns: os.remove(name)

    def checkFM(inc, az):
        #Make temporary dir for match image
        os.mkdir(f"{workdir}/{m3id}")
        #Create a hillshade map from the topography
        gdal.DEMProcessing(f"{workdir}/{m3id}_hillshade_az{az:0.2f}_inc{inc:0.2f}.tif", f"{workdir}/{m3id}_topo_sinu.tif", "hillshade",
                    options=gdal.DEMProcessingOptions(
                        format='GTiff',
                        alg='ZevenbergenThorne',
                        azimuth=az,
                        altitude=inc
                    ))

        #Define files for RDN, hillshade, match output
        inrdn_fm = f'{workdir}/{m3id}_RDN_average_byte.tif'
        inshd_fm = f'{workdir}/{m3id}_hillshade_az{az:0.2f}_inc{inc:0.2f}.tif'
        out_fm = f'{workdir}/{m3id}/{m3id}_az{az:0.2f}_inc{inc:0.2f}'
        #Run image match
        try:
            match_images(inrdn_fm, inshd_fm, out_fm)
        except Exception as e:
            print(e)
            print("MATCH FAILED")
            shutil.rmtree(f"{workdir}/{m3id}")
            return False

        #Return True if process worked, else remove the temp dir and return False
        if os.path.isfile(f"{out_fm}.png"):
            return True
        else:
            shutil.rmtree(f"{workdir}/{m3id}")
            return False
    
    # Check if a match was successful
    matched = checkFM(incmean, azmmean)
    infodict['FIRST_TRY']=matched
    if not matched:
        # Retry matches for azm=[azmmean-60, azmmean+60] and 
        # inc = [10,80] in intervals of 10
        print("RETRYING MATCH")
        azm, inc = np.meshgrid(np.arange(azmmean-60, azmmean+61, 10),
                               np.arange( 10, 81, 10))
        coords = np.vstack((inc.flatten(), azm.flatten())).T
        for k in range(len(coords)):
            if checkFM(*coords[k]):
                matched = True
                infodict['INC_MATCH'] = coords[k][0]
                infodict['AZM_MATCH'] = coords[k][1]
                break
    else:
        infodict['INC_MATCH'] = infodict['INC']
        infodict['AZM_MATCH'] = infodict['AZM']
    # If there has been any match, move the matching directory to Results/Worked, else 
    # write a file to Results/Failed indicating no match was found.
    if matched:
        shutil.move(f"{workdir}/{m3id}", f"Results/Worked/{m3id}")
        shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"Results/Worked/{m3id}/{m3id}_RDN_average_byte.tif")
    else:
        shutil.move(f"{workdir}/{m3id}_RDN_average_byte.tif", f"Results/Failed/{m3id}_RDN_average_byte.tif")
    infodict['WORKED'] = matched
    shutil.rmtree(workdir)
    return infodict


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(run_match(sys.argv[1]))
        quit()
    if sys.argv[1] == '-f':
        fnout = 'dataout.csv' if len(sys.argv) < 4 else sys.argv[3]
        m3ids = open(sys.argv[2], 'r').read().split('\n')

        if fnout not in os.listdir('Results'):
            data = pd.DataFrame(columns=DF_COLS)
        else:
            data = pd.read_csv(f'Results/{fnout}')
        
        for k_m3id in range(len(m3ids)):
            if m3ids[k_m3id] in data['M3ID'].values:
                print(f"Skipping {m3ids[k_m3id]} - MATCH {'WORKED' if data[data['M3ID'] == m3ids[k_m3id]]['WORKED'].values[0] else 'FAILED'}")
                continue
            k_data = run_match(m3ids[k_m3id])
            data = pd.concat([data, pd.DataFrame(k_data, index=[0])])
            data.to_csv(f'Results/{fnout}',index=False)
    else:
        print("INVALID PARAMS")
        print("SINGLE RUN:\t python LTB_FM_M3.py <M3ID>")
        print("BATCH RUN:\t python LTB_FM_M3.py -f <M3 list file>")
    

