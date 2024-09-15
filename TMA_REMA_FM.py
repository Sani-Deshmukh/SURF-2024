import os
import cv2
import numpy as np
import shutil
import pandas as pd
from osgeo import gdal, osr, gdalconst, ogr
from FM_toolkit import *
import click

#CONSTANTS
#change this to match our project
DF_COLS = ['TMA_ID', 'WORKED',
           'FIRST_TRY', 'FILE_FOUND',
           'LON', 'LON_MIN', 'LON_MAX',
           'LAT', 'LAT_MIN', 'LAT_MAX',
           'P_X', 'P_X_MIN', 'P_X_MAX',
           'P_Y', 'P_Y_MIN', 'P_Y_MAX',
           'REPR_LON', 'REPR_LON_MIN', 'REPR_LON_MAX',
           'REPR_LAT', 'REPR_LAT_MIN', 'REPR_LAT_MAX',
           'INC', 'AZM', 'INC_MATCH', 'AZM_MATCH']
RESULTS_PATH = "../Results"
WORKED_PATH = "../Results/Worked"
FAILED_PATH = "../Results/Failed"
REMA_PATH = "../Results/Rema"

# Determines the size of the bounding match we are attempting to match on:
    # Bugger teh bounding box, the better
ALTITUDE_CROP_BUFFER = {25000: 3000, 18000: 4000, 10000: 4000}
CROP_BUFFER = 4000
# Change this to make left bigger again
X_LEFT_BORDER_CROP = 0.8
X_RIGHT_BORDER_CROP = 0.8
Y_BORDER_CROP = 0.7

REMA_PATH = '/Volumes/pgc/data/elev/dem/setsm/REMA/release_staging/mosaics/v2.0/pgc/rema_2_0_2m_dem_tiles_relative.vrt'
REMA = gdal.Open(REMA_PATH, gdal.GA_ReadOnly)

FEET_TO_METERS = 0.3048


def checkFM(inc, azm, dem, tma, workdir, tma_id):
    # Output filepath for the hill shade
    inshd_fm = f'{workdir}/{tma_id}_hillshade_az{azm:0.2f}_inc{inc:0.2f}.tif'

    # base path used to check successful matches
    out_fm = f'{workdir}/{tma_id}/{tma_id}_az{azm:0.2f}_inc{inc:0.2f}'

    # Make temporary dir for match image
    if not os.path.exists(f"{workdir}/{tma_id}"):
        os.mkdir(f"{workdir}/{tma_id}")

    # Create a hillshade map from the topography
        # Because different constrast hillshades needed for different types of images,
        # used average color values of aerial image to determine contrast: darker images = low contrast
    contrast_rema = 1
    aerial_img = cv2.imread(tma, cv2.IMREAD_GRAYSCALE)
    average_color = np.mean(aerial_img)
    print("AVERAGE COLOR: ", average_color)
    if average_color > 150:
        contrast_rema = 4
    gdal.DEMProcessing(inshd_fm, dem,
                       "hillshade",
                       options=gdal.DEMProcessingOptions(
                           format='GTiff',
                           alg='ZevenbergenThorne',
                           azimuth=azm,
                           altitude=inc,
                           # zFactor to increase contrast
                           zFactor=contrast_rema
                       ))
    try:
        # What exactly is out_fm ????
        match_images(tma, inshd_fm, out_fm)
        print("WORKED!! :)))")
    except Exception as e:
        print(e)
        print("MATCH FAILED for ", tma_id, ":(")
        # shutil.rmtree(f"{workdir}/{tma_id}")
        return False

    # Return True if process worked, else remove the temp dir and return False
    if os.path.isfile(f"{out_fm}.png"):
        return True
        print("found the png")
    else:
        shutil.rmtree(f"{workdir}/{tma_id}")
        return False


def run_match(tma_id):
    '''
          Runs feature matching for a given tma ID
    '''

    # Verify all the output and source folders in directory
    if not (os.path.exists(RESULTS_PATH)):
        os.makedirs(RESULTS_PATH)
    if not (os.path.exists(WORKED_PATH)):
        os.makedirs(WORKED_PATH)
    if not (os.path.exists(FAILED_PATH)):
        os.makedirs(FAILED_PATH)

    # Create a directory to run all of the processes/make files in.
    workdir = f'{tma_id}_work'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    # Prepare CSVs for image data
    df = pd.read_csv('usgs_index_aerial_tma_pt_test.csv')
    pa = pd.read_csv('usgs_index_aerial_tma_photo_attributes.csv', dtype={'column_name': 'object'})

    # Prepare the images for border cropping
    flight_id = int(tma_id[2:6], 10)

    # Getting the path to the image for given tma_id
    img_path = os.path.join(tma_id + ".tif")
    print(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    center = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    retain_width_l = int(img.shape[1] * X_LEFT_BORDER_CROP)
    retain_width_r = int(img.shape[1] * X_RIGHT_BORDER_CROP)
    retain_height = int(img.shape[0] * Y_BORDER_CROP)

    # Calculate the borders to crop
    x_start = center[0] - retain_width_l // 2
    x_end = center[0] + retain_width_r // 2
    y_start = center[1] - retain_height // 2
    y_end = center[1] + retain_height // 2

    # Ensure that the coordinates are within the image dimensions
    x_start = max(0, x_start)
    x_end = min(img.shape[1], x_end)
    y_start = max(0, y_start)
    y_end = min(img.shape[0], y_end)

    # Crop the image
    cropped_image = img[y_start:y_end, x_start:x_end]

    cropped_path = f'{workdir}/{tma_id}_cropped.tif'
    cv2.imwrite(cropped_path, cropped_image)

    # Getting Image coordinates
    try:
        # Filter DataFrame based on both acq_id and photo_num
        # Extract photo number
        photo_str = img_path.split('V')[1].split('.')[0]
        photo = int(photo_str, 10)
        row = df.loc[(df['tma_num'] == flight_id) & (df['photo_num'] == photo)]
        row_pa = pa.loc[pa['image_id'] == tma_id]
        if not row.empty:
            lat = row['lat'].values[0]
            lon = row['lon'].values[0]
            print("COOR BEFORE CONVERSION: ", lat, lon)
            alt = row['alt_ft'].values[0]
            # print(f"{image} Latitude: {lat}, Longitude: {lon}")
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(lon, lat)

            # Create a spatial reference object with EPSG:4326 (WGS84)
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(4326)  # EPSG:4326 is WGS84
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(3031)  # EPSG:3857 is Polar Stenographic

            # Create a transformation object from src_srs to dst_srs
            transform = osr.CoordinateTransformation(src_srs, dst_srs)
            transformed_point = transform.TransformPoint(lat, lon)
            x_meters, y_meters, z_meters = transformed_point
            print(tma_id, "transformed coordinates", x_meters, y_meters, z_meters, alt)
            if alt > 25000:
                buffer = ALTITUDE_CROP_BUFFER[25000]
            elif alt > 18000:
                buffer = ALTITUDE_CROP_BUFFER[18000]
                print("this done correct")
            elif alt >= 9000:
                buffer = ALTITUDE_CROP_BUFFER[10000]
            else:
                # Default altitude when no altitude found in the csv
                buffer = ALTITUDE_CROP_BUFFER[25000]
            # Clipping REMA
            rema_subset = clip_rema(x_meters, y_meters, tma_id, buffer, workdir)

            # Calling for checking match on potential hill shades
            matches_found = 0
            # for inc in range(90, 10, -10):
            for inc in range(60, 30, -10):
                # for azm in range(-180, 180, 10):
                for azm in range(120, 170, 10):
                    print("AZM/INC", azm, inc)
                    if checkFM(inc, azm, rema_subset, cropped_path, workdir, tma_id):
                        matches_found += 1
                        print("Matches found: ", matches_found)
                        # break

            print("Check FM iterations complete")

    except Exception as e:
        print(e)


def clip_rema(x_meters, y_meters, tma_id, buffer, workdir):
    # Define the path for the output topography file
    print("buffer", buffer)
    topo_path = f'{workdir}/{tma_id}_topo.tif'
    bbox = (x_meters - buffer, y_meters - buffer, x_meters + buffer, y_meters + buffer)
    # Warp the DEM using the transformed coordinates
    gdal.Warp(topo_path, REMA,
              options=gdal.WarpOptions(
                  format='GTiff',
                  outputBounds=bbox,
                  resampleAlg='bilinear',
              ))
    print(bbox)
    return topo_path


if __name__ == "__main__":
    # argument handling

    # Use click
    if len(sys.argv) == 2:
        print(run_match(sys.argv[1]))
        quit()

    if sys.argv[1] == '-f':
        fnout = 'dataout.csv' if len(sys.argv) < 4 else sys.argv[3]
        tma_ids = open(sys.argv[2], 'r').read().split('\n')

        if fnout not in os.listdir('Results'):
            data = pd.DataFrame(columns=DF_COLS)
        else:
            data = pd.read_csv(f'Results/{fnout}')

        for k_m3id in range(len(tma_ids)):
            if tma_ids[k_m3id] in data['TMA_ID'].values:
                print(
                    f"Skipping {tma_ids[k_m3id]} - MATCH {'WORKED' if data[data['TMA_ID'] == tma_ids[k_m3id]]['WORKED'].values[0] else 'FAILED'}")
                continue
            k_data = run_match(tma_ids[k_m3id])
            data = pd.concat([data, pd.DataFrame(k_data, index=[0])])
            data.to_csv(f'Results/{fnout}', index=False)
    else:
        print("INVALID PARAMS")
        print("SINGLE RUN:\t python LTB_FM_M3.py <TMA_ID>")
        print("BATCH RUN:\t python LTB_FM_M3.py -f <M3 list file>")
