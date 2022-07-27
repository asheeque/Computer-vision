#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

from cv2 import imshow
import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    img1c = img1
    img2c = img2
    img1 = cv2.copyMakeBorder(img1, 200, 200, 200, 200, cv2.BORDER_CONSTANT)
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_grey, None)
    kp2, des2 = sift.detectAndCompute(img2_grey, None)

    bestMatches = getBestMatch(des1,des2)

    src_pts = np.float32([ kp1[m.get("imgOneIdx")].pt for m in bestMatches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.get("imgTwoIdx")].pt for m in bestMatches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)

    warped = cv2.warpPerspective(img2, M, ((img2.shape[1] + img1.shape[1]), img1.shape[0]))

    img2_1 = warped[0:img1.shape[0], 0:img1.shape[1]]
    stitched_image = final(img2_1,img1)
   
    cv2.imwrite(savepath, stitched_image)
    return 

def final(img1, img2):
    new = img2
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if abs(np.sum(img1[i][j])) == abs(np.sum(img2[i][j])):
                continue
            elif abs(np.sum(img1[i][j])) > abs(np.sum(img2[i][j])):
                new[i][j] = img1[i][j]
            elif abs(np.sum(img1[i][j])) < abs(np.sum(img2[i][j])):
                new[i][j] = img2[i][j]
    return new


def getBestMatch(imgOneFeatures,imgTwoFeatures):

    res = []
    for i in range(len(imgOneFeatures)):

        singleImgOneFeature = imgOneFeatures[i]
        ssdValues = doSSD(singleImgOneFeature,imgTwoFeatures)
        # print(ssdValues.shape)
        sortedssdValues = np.sort(ssdValues)
        firstMatch = sortedssdValues[0]
        secondMatch = sortedssdValues[1]
        if firstMatch < 0.8 * secondMatch:
            firstMatchIdx = np.where(ssdValues == firstMatch)[0][0]
            match = {
                "imgOneIdx":i,
                "imgTwoIdx":firstMatchIdx,
                "distance":firstMatch
            }
            res.append(match)
    return res


def doSSD(singleImgOneFeature,imgTwoFeatures):

    x = np.subtract(singleImgOneFeature,imgTwoFeatures)
    x = np.square(x)
    x = np.sum(x,axis = 1)
    x = np.sqrt(x)
    return x

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

