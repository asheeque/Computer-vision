# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    imageArr = imgs.copy()
    oneHotEncoding = np.zeros((N,N))
    oneHotEncoding = check_overlap(imageArr,oneHotEncoding,N)
    print(oneHotEncoding)
    overlap_arr = oneHotEncoding.copy()
    final = stitch_panorama(imgs,oneHotEncoding,N)
    cv2.imwrite('result.png', final)
    return overlap_arr

def check_overlap(imageArr,oneHotEncoding,N):

    for i in range(N):
        for j in range(N):
            if i == j:  
                oneHotEncoding[i][j] = 1
            else:
                imageArr[j] = cv2.copyMakeBorder(imageArr[j], 100, 100, 100, 100, cv2.BORDER_CONSTANT)
                warped, twoWarped, lenBestMatch,imgTwoOneWarped = getWarpedImage(imageArr[i], imageArr[j])
                if lenBestMatch < 300:
                    continue
                overlap_val = get_overlap_value(imageArr[i], twoWarped)
                if overlap_val > 2.5:
                    oneHotEncoding[i][j],oneHotEncoding[j][i] = 1,1
    return oneHotEncoding

def stitch_panorama(imageArr,oneHotEncoding,N):
    one_hot = oneHotEncoding
    final_image = imageArr[0]
    for i in range(len(one_hot)):
        for j in range(len(one_hot)):
            if i ==j :
                continue
            if one_hot[i][j] == 1:
                final_image  = cv2.copyMakeBorder(final_image , 100, 100, 100, 100, cv2.BORDER_CONSTANT)
                warped, twoWarped, lenBestMatch,imgTwoOneWarped = getWarpedImage(final_image, imageArr[j])
                if lenBestMatch < 300:
                    warped, twoWarped, lenBestMatch,imgTwoOneWarped = getWarpedImage( imageArr[j],final_image)
                    if lenBestMatch < 300:
                        continue
                final_image = final(imgTwoOneWarped, warped)
                one_hot[j][i] = 0
    return final_image


def get_overlap_value(imageOne, imageTwo):
    overlappingPixels = 0
    totalPixels = 0
    for i in range(imageOne.shape[0]):
        for j in range(imageOne.shape[1]):
            if abs(np.sum(imageOne[i][j])) == abs(np.sum(imageTwo[i][j])) and abs(np.sum(imageOne[i][j])) == 0:
                continue
            elif abs(np.sum(imageOne[i][j])) == abs(np.sum(imageTwo[i][j])):
                overlappingPixels = overlappingPixels + 1
                totalPixels = totalPixels + 1
            else:
                totalPixels = totalPixels + 1
    overlap_val = (overlappingPixels/totalPixels)*100
    print(overlap_val)
    return overlap_val


def getWarpedImage(imageOne, imageTwo):
    img1_grey = cv2.cvtColor(imageOne, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(imageTwo, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_grey, None)
    kp2, des2 = sift.detectAndCompute(img2_grey, None)
    bestMatches = getBestMatch(des1,des2)

    src_pts = np.float32([ kp1[m.get("imgOneIdx")].pt for m in bestMatches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.get("imgTwoIdx")].pt for m in bestMatches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)

    warped = cv2.warpPerspective(imageTwo, M, ((imageTwo.shape[1] + imageOne.shape[1]), imageOne.shape[0]))
    img2_1_warped = warped[0:imageOne.shape[0], 0:imageOne.shape[1]]
    warped_img1 = warped.copy()
    warped_img1[0:imageOne.shape[0], 0:imageOne.shape[1]] = imageOne
    lenBestMatch = len(bestMatches)

    return  warped_img1,warped, lenBestMatch,img2_1_warped



def getBestMatch(imgOneFeatures,imgTwoFeatures):

    res = []
    for i in range(len(imgOneFeatures)):

        singleImgOneFeature = imgOneFeatures[i]
        ssdValues = doSSD(singleImgOneFeature,imgTwoFeatures)
        # print(ssdValues.shape)
        sortedssdValues = np.sort(ssdValues)
        firstMatch = sortedssdValues[0]
        secondMatch = sortedssdValues[1]
        
        # firstMatchIdx = np.searchsorted(ssdValues,firstMatch)
        if firstMatch < 0.85 * secondMatch:
            firstMatchIdx = np.where(ssdValues == firstMatch)[0][0]
            match = {
                "imgOneIdx":i,
                "imgTwoIdx":firstMatchIdx,
                "distance":firstMatch
            }
            res.append(match)
        
    return res

def final(imageOne, imageTwo):
    
    new = imageTwo
    for i in range(imageOne.shape[0]):
        for j in range(imageOne.shape[1]):
            if abs(np.sum(imageOne[i][j])) == abs(np.sum(imageTwo[i][j])):
                continue
            elif abs(np.sum(imageOne[i][j])) > abs(np.sum(imageTwo[i][j])):
                new[i][j] = imageOne[i][j]
            elif abs(np.sum(imageOne[i][j])) < abs(np.sum(imageTwo[i][j])):
                new[i][j] = imageTwo[i][j]
    return new

def doSSD(singleImgOneFeature,imgTwoFeatures):

    x = np.subtract(singleImgOneFeature,imgTwoFeatures)
    x = np.square(x)
    x = np.sum(x,axis = 1)
    x = np.sqrt(x)
    return x


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('result.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # #bonus
    # overlap_arr2 = stitch('t3', savepath='task3.png')
    # with open('t3_overlap.txt', 'w') as outfile:
    #     json.dump(overlap_arr2.tolist(), outfile)
