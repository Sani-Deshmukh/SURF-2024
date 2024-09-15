import cv2 as cv
import numpy as np

class SIFTDetector:


    def __init__(self, nFeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=13, sigma=1.6):
        # Defining SIFT parameters
        '''
        :param nFeatures: number of features to retain (0 being all features detected)
        :param nOctaveLayers: number of layers in each octave of Gaussian pyramid
        :param contrastThreshold: contrast thresh used to filter weak features in low contrtast
        :param edgeThreshold: threshold used to filter out edge like features
        :param sigma: initial image smoothing (sigma of the gaussian applied to the input image @ octave 0
        '''
        self.nFeatures = nFeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

        # Initializing SIFT detector
        self.sift = cv.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)


    def detect(self, im):
        # Convert input image to gray scale
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        # Detect keypoints and compute descriptors
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des

class AKAZEDetector:
    def __init__(self, descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3,
                 threshold=0.0001, nOctaves=8, nOctaveLayers=8, diffusivity=cv.KAZE_DIFF_PM_G1):
        '''
        :param descriptor_type: Type of descriptor used by AKAZE.
        :param descriptor_size: Size of the descriptor. Typically, values are 0 (default) for the automatic size or
        specific sizes such as 32 or 64.
        :param descriptor_channels: Number of channels in the descriptor. Generally, this is 1 (grayscale) or 3 (color).
        :param threshold: Threshold for detecting keypoints. Lower values detect more keypoints, while higher values
        detect fewer. Typical values range from 0.001 to 0.01.
        :param nOctaves: Number of octaves in the image pyramid used for detection. Values generally range from 4 to 8.
        :param nOctaveLayers: Number of layers per octave in the pyramid. Values typically range from 1 to 8.
        :param diffusivity: Type of diffusivity function used. Options include:
            - cv.KAZE_DIFF_CONTRAST: Contrast diffusivity (default)
            - cv.KAZE_DIFF_PM_G1: Perona-Malik diffusivity function (G1)
            - cv.KAZE_DIFF_PM_G2: Perona-Malik diffusivity function (G2)
        '''
        self.descriptor_type = descriptor_type
        self.descriptor_size = descriptor_size
        self.descriptor_channels = descriptor_channels
        self.threshold = threshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.diffusivity = diffusivity

        self.akaze = cv.AKAZE_create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity)

    # Main matches function to call on the class of detector
    def detect(self, im):
        # Convert input image to gray scale
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = self.akaze.detectAndCompute(gray, None)
        return kp, des

class ORBDetector:
    '''
    :param: nfeatures: The maximum number of features to retain.
    :param: scaleFactor: Pyramid decimation ratio, greater than 1.
    :param: nlevels: The number of pyramid levels.
    :param: edgeThreshold: The size of the border where the features are not detected.
    :param: firstLevel: The level of the pyramid to put the source image to.
    :param: WTA_K: The number of points that produce each element of the oriented BRIEF descriptor.
    :param: scoreType: The algorithm to rank features (e.g., cv.ORB_HARRIS_SCORE).
    :param: patchSize: The size of the patch used by the oriented BRIEF descriptor.
    :param: fastThreshold: The FAST threshold for detecting keypoints.
    '''
    def __init__(self, nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                 scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20):
        # Initializing ORB detector with specified parameters
        self.orb = cv.ORB_create(nfeatures=nfeatures,
                                 scaleFactor=scaleFactor,
                                 nlevels=nlevels,
                                 edgeThreshold=edgeThreshold,
                                 firstLevel=firstLevel,
                                 WTA_K=WTA_K,
                                 scoreType=scoreType,
                                 patchSize=patchSize,
                                 fastThreshold=fastThreshold)

    def detect(self, im):
        # Convert input image to grayscale
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des


class KAZEDetector:
    def __init__(self, extended=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_PM_G2):
        # Initializing KAZE detector with specified parameters
        self.extended = extended
        self.threshold = threshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.diffusivity = diffusivity
        self.kaze = cv.KAZE_create(extended=self.extended,
                                    threshold=self.threshold,
                                    nOctaves=self.nOctaves,
                                    nOctaveLayers=self.nOctaveLayers,
                                    diffusivity=self.diffusivity)

    def detect(self, im):
        # Convert input image to grayscale
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = self.kaze.detectAndCompute(gray, None)

        # Ensure descriptors are of type CV_32F
        if des is not None and des.dtype != np.float32:
            des = des.astype(np.float32)

        return kp, des


# class SURFDetector:
#     def __init__(self, descriptor_type=cv.KAZE_DESCRIPTOR_MLDB):
#
# class BRISKDetector:
#     def __init__(self, descriptor_type=cv.BRISK_SIFT):
#
# class BRIEFDetector:
#     def __init__(self, descriptor_type=cv.BRIEF_SIFT):
#
#
#
#
