import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def colorTransfer(source, dest):
    # Transfers the tone of source to dest
    source_gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    dest_gray = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)

    s_mean, s_std = cv.meanStdDev(source_gray, mask=cv.inRange(source_gray, 0, 254))
    d_mean, d_std = cv.meanStdDev(dest_gray, mask=cv.inRange(dest_gray, 0, 254))

    s_mean, s_std = np.hstack(s_mean)[0], np.hstack(s_std)[0]
    d_mean, d_std = np.hstack(d_mean)[0], np.hstack(d_std)[0]

    return cv.cvtColor(
            np.clip(((dest_gray-d_mean)*(s_std/d_std))+s_mean, 0, 255).astype(np.uint8),
            cv.COLOR_GRAY2BGR)

class IterativeFBM:
    def __init__(self, detector, matcher):
        # Defining FBM parameters
        self.detector = detector
        self.matcher = matcher
        self.maxIterations = 15
        # Default 1.5, the reprojection tolerance determines how much distortion is allowed when
        # projecting the keypoints between the model and the aerial image.
        self.reprojTolerance = 1.5
        # Changed RANSAC (Outlier Detection threshold from 3 )
        # Needed to improve the outlier detection: reduced to 2
        self.redundancyTolerance = 2.

    def reprojTest(self, ptA, ptB, homography):
        # Tests the reprojection of ptA to ptB using homography
        # Returns True if the reprojection is within tolerance of ptB
        ptA = cv.perspectiveTransform(
                np.array([[ptA]], dtype=np.float64),
                homography
                )[0,0]
        ptB = np.array(ptB)

        dPt = ptA - ptB
        maxSqr = self.reprojTolerance*self.reprojTolerance
        distSqr = dPt[0]*dPt[0] + dPt[1]*dPt[1]

        return distSqr <= maxSqr

    def redundancyCheck(self, matches):
        # Checks if there are redundant matches based off of the match index on the ground truth
        # Returns the unique matches and retains the best match our of redundant matches
        uniqueMatches = []
        for i, matchA in enumerate(matches):
            isMin = True
            for j, matchB in enumerate(matches, start=i+1):
                if matchA.trainIdx == matchB.trainIdx and matchA.distance > matchB.distance:
                    isMin = False
                    break
            if isMin:
                uniqueMatches.append(matchA)
        return uniqueMatches

    def step(self, imA, imB, M, kpB, desB):
        imAprime = cv.warpPerspective(imA, M, imA.shape[:-1][::-1])
        kpAprime, desAprime = self.detector.detect(imAprime)
        matches = self.matcher.match(desAprime, desB)
        # Find homography

        ptsA, ptsB = [], []
        for match in matches:
            ptsA.append(kpAprime[match.queryIdx].pt)
            ptsB.append(kpB[match.trainIdx].pt)

        try:
            homography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB), cv.RANSAC, self.redundancyTolerance)
        except:
            try:
                homography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB)) # Default to least-squares
            except Exception as e:
                assert len(ptsA) >= 4, f'Less than 4 matches found for homography! + MATCHES: {len(ptsA)}'
                raise e
        M = homography.dot(M)

        # Test points against reprojection
        matches = list(filter(lambda match:
            self.reprojTest(
                kpAprime[match.queryIdx].pt, kpB[match.trainIdx].pt, homography),
            matches))

        return (M, matches, kpAprime, desAprime)


    def match(self, imA, imB):
        MINIMUM_MATCHES = 4
        # Color transfer
        imA = colorTransfer(imB, imA)

        # Match image heights
        height = min(imA.shape[0], imB.shape[0])
        imA = cv.resize(imA, dsize=(round((height/imA.shape[0])*imA.shape[1]), height))
        imB = cv.resize(imB, dsize=(round((height/imB.shape[0])*imB.shape[1]), height))

        # Initialize data
        kpB, desB = self.detector.detect(imB)

        chaoticHomography = np.identity(3)
        orderlyHomography = np.identity(3)
        orderlyKeypoints = []
        orderlyDescriptors = []
        orderlyMatches = []

        # Iteration
        i = 0
        while i <= self.maxIterations:
            i = i + 1

            # Chaotic step
            prevChaoticHomography = chaoticHomography.copy()
            try:
                (chaoticHomography, chaoticMatches,
                        kpAprime, desAprime) = self.step(imA, imB, chaoticHomography, kpB, desB)
            except Exception as e:
                if i == 1: raise e
                break

            # Redundancy check
            for match in chaoticMatches:
                orderlyMatches.append(cv.DMatch(
                    match.queryIdx + len(orderlyKeypoints), match.trainIdx, match.distance))
            orderlyMatches = self.redundancyCheck(orderlyMatches)

            # Append chaotic data
            for kp in kpAprime:
                # Invert keypoints to original image space
                kp.pt = tuple(
                        cv.perspectiveTransform(
                            np.array([[kp.pt]], dtype=np.float64),
                            np.linalg.inv(prevChaoticHomography)
                            )[0,0])
            chaoticKeypoints = list(kpAprime)
            chaoticDescriptors = list(desAprime)
            orderlyKeypoints = orderlyKeypoints + chaoticKeypoints
            orderlyDescriptors = orderlyDescriptors + orderlyKeypoints

            if len(orderlyMatches) > 0:
                # Find orderly homography
                ptsA, ptsB = [], []
                for match in orderlyMatches:
                    ptsA.append(orderlyKeypoints[match.queryIdx].pt)
                    ptsB.append(kpB[match.trainIdx].pt)
                try:
                    # Attempt to use RHO: An alternate outlier detection algorithm. This step detects outliers in the
                    # orderly step of matching (happens after chaotic has found a few keypoints to begin matching)
                    # In other words, if the second image in the output png is more distorted, reduce outlier
                    # detection here
                    orderlyHomography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB), cv.RHO, 2)
                    print("RASNAC succesful")
                except:
                    try:
                        # Default to least-squares if there aren't enough points
                        orderlyHomography, mask = cv.findHomography(np.array(ptsA), np.array(ptsB))
                    except Exception as e:
                        if i > 1: break
                        assert len(ptsA) >= MINIMUM_MATCHES, f'WRONG ONE Less than 4 matches found for homography! MATCHES: {len(ptsA)}'
                        raise e

                # Orderly step
                prevOrderlyHomography = orderlyHomography.copy()
                try:
                    (orderlyHomography, newOrderlyMatches,
                            kpAprime, desAprime) = self.step(imA, imB, orderlyHomography, kpB, desB)
                except Exception as e:
                    if i == 1: raise e
                    break

                # Redundancy check
                for match in newOrderlyMatches:
                    orderlyMatches.append(cv.DMatch(
                        match.queryIdx + len(orderlyKeypoints), match.trainIdx, match.distance))
                orderlyMatches = self.redundancyCheck(orderlyMatches)

                # Append orderly data
                for kp in kpAprime:
                    # Invert keypoints to original image space
                    kp.pt = tuple(
                            cv.perspectiveTransform(
                                np.array([[kp.pt]], dtype=np.float64),
                                np.linalg.inv(prevOrderlyHomography)
                                )[0,0])
                orderlyKeypoints = orderlyKeypoints + list(kpAprime)
                orderlyDescriptors = orderlyDescriptors + list(desAprime)
        assert len(orderlyMatches) > 0, 'No matches found during iterative FBM!'

        return (chaoticHomography, orderlyHomography,
                chaoticKeypoints, orderlyKeypoints,
                chaoticDescriptors, orderlyDescriptors,
                kpB, desB,
                chaoticMatches, orderlyMatches,
                imA, imB, i)

    # very very simple matching test
    def quick_match (self, imA, imB):
        # Color transfer
        imA = colorTransfer(imB, imA)

        # Match image heights
        height = min(imA.shape[0], imB.shape[0])
        imA = cv.resize(imA, dsize=(round((height / imA.shape[0]) * imA.shape[1]), height))
        imB = cv.resize(imB, dsize=(round((height / imB.shape[0]) * imB.shape[1]), height))

        # Initialize data
        kpA, desA = self.detector.detect(imA)
        kpB, desB = self.detector.detect(imB)

        #match descriptors
        matches = self.matcher.match(desA, desB)

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches
        result_img = cv.drawMatches(imA, kpA, imB, kpB, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite("test_match_out.png", result_img)

        return result_img
