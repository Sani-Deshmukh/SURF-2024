#This code matches features between M3 data and LOLA hillshade data.
#This script calls upon detector and feature based matching scripts that can be found here: ../src/lunarreg/
#Those scripts are loaded as modules within lunarreg in this code.
#This is only the code that calls those scripts and generates outputs. Changes to the actual feature matching and detecting should be done on those scripts.

#The code expects three arguments: 1) a radiance file, 2) a hillshade file and 3) a name for the output files.
#The code is called by using python, i.e.: python M3-LOLA.py in-radiance.tif in-hillshade.tif outfile

import sys

import cv2

sys.path.append('../LunarReg/src')
from os import listdir

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from lunarreg import IterativeFBM
from lunarreg.detectors import SIFTDetector
from lunarreg.matchers import BFMatcher
from lunarreg.detectors import AKAZEDetector
from lunarreg.detectors import ORBDetector
from lunarreg.detectors import KAZEDetector

#This function records a CSV file with match coordinates and saves a PNG file of the image-to-image plot
def record_plot(
        imA, imB,
        chaoticKeypoints, orderlyKeypoints, kpB,
        chaoticMatches, orderlyMatches,
        chaoticHomography, orderlyHomography, output):
    
    fig, axs = plt.subplots(3,2,figsize=(3,15))

    imAprimeC = cv.warpPerspective(imA, chaoticHomography, imA.shape[:-1][::-1])
    imAprimeO = cv.warpPerspective(imA, orderlyHomography, imA.shape[:-1][::-1])
    
    imgs = (imA, imB, imAprimeC, imB, imAprimeO, imB)

    for axidx in range(len(axs.flatten())):
        axs.flatten()[axidx].imshow(imgs[axidx])
        axs.flatten()[axidx].axis('off')

    #Create a CSV file and write a header line with the column names
    f = open(output + ".csv", "a")
    f.write("HillshadeColumn,Hillsm hadeRow,TMAColumn,TMARow\n")

    for match in chaoticMatches:
        ptA = tuple(
                cv.perspectiveTransform(
                    np.array([[chaoticKeypoints[match.queryIdx].pt]], dtype=np.float64),
                    chaoticHomography)[0, 0])
        con = ConnectionPatch(
                xyA=ptA, xyB=kpB[match.trainIdx].pt,
                coordsA='data', coordsB='data',
                axesA=axs[1,0], axesB=axs[1,1], 
                color='red', linewidth=.1)
        axs[1,1].add_artist(con)

    for match in orderlyMatches:
        ptA = tuple(
                cv.perspectiveTransform(
                    np.array([[orderlyKeypoints[match.queryIdx].pt]], dtype=np.float64),
                    orderlyHomography)[0,0])

        #Write the match coordinates for the orderly matches to the CSV file
        f.write(str(ptA[0]) + "," + str(ptA[1]) + "," + str(kpB[match.trainIdx].pt[0]) + "," + str(kpB[match.trainIdx].pt[1]) + "\n")

        con = ConnectionPatch(
                xyA=ptA, xyB=kpB[match.trainIdx].pt,
                coordsA='data', coordsB='data',
                axesA=axs[2,0], axesB=axs[2,1], color='red', linewidth=.1)
        axs[2,1].add_artist(con)

    plt.savefig(output + '.png', dpi=1200)
    f.close()

def match_images(A, B, ofn):
    imA = cv.imread(A)
    imB = cv.imread(B)
    # # Load detectors and matchers
    detector = AKAZEDetector()
    # detector = KAZEDetector()
    # detector = ORBDetector()
    # detector = SIFTDetector(nOctaveLayers=8, contrastThreshold=0.02, sigma=1.8)
    matcher = BFMatcher()
    matcher.ratio = 0.65 # originally 0.65

    # Load iterative FBM
    fbm = IterativeFBM(detector, matcher)
    fbm.maxIterations = 15 # /=10

    (chaoticHomography,     orderlyHomography,
     chaoticKeypoints,      orderlyKeypoints,
     chaoticDescriptors,    orderlyDescriptors,
     kpB,                   desB,
     chaoticMatches,        orderlyMatches,
     imA, imB, i) = fbm.match(imA, imB)
    
    record_plot(imA, imB,
                chaoticKeypoints, orderlyKeypoints, kpB,
                chaoticMatches, orderlyMatches, 
                chaoticHomography, orderlyHomography, ofn)