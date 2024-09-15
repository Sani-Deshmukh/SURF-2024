import cv2 as cv
import numpy as np

def ratioTest(knnMatches, ratio):
    # Lowe's ratio test to discard insufficiently strong matches
    return list(filter(lambda match: match[0].distance <= match[1].distance*ratio, knnMatches))

def symmetryTest(knnMatchesA, knnMatchesB):
    # Discard matches that are not symmetric
    matches = []
    for matchA in knnMatchesA:
        for matchB in knnMatchesB:
            if matchA[0].queryIdx == matchB[0].trainIdx and matchA[0].trainIdx == matchB[0].queryIdx:
                matches.append(cv.DMatch(matchA[0].queryIdx, matchA[0].trainIdx, matchA[0].distance))
                # Addded this break for some reason
                break
    return matches

class BFMatcher:
    def __init__(self):
        # ## USE FOR SIFT
        # normType = cv.NORM_L2
        # crossCheck = False
        # self.bf = cv.BFMatcher(normType, crossCheck=crossCheck)
        # self.ratio = 0.65

        ## USE FOR AKAZE
        normType = cv.NORM_HAMMING
        self.bf = cv.BFMatcher(normType)
        # this ratio determines how lenient the matches can be with lower ratio
        # being more stricter matches. Originally 0.65, tried multiple values ranging from 0.6 to 0.8
        self.ratio = 0.75

        ## USE FOR ORB
        # normType = cv.NORM_HAMMING
        # self.bf = cv.BFMatcher(normType)
        # self.ratio = 0.75

    def match(self, desA, desB):
        # Ratio test using 2 nearest neighbors
        knnMatchesA = ratioTest(self.bf.knnMatch(desA, desB, 2), self.ratio)
        knnMatchesB = ratioTest(self.bf.knnMatch(desB, desA, 2), self.ratio)

        # Symmetry test
        matches = symmetryTest(knnMatchesA, knnMatchesB)

        return matches