'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition
'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''
import dlib
from sklearn.cluster import KMeans

def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    image_arr = get_images(input_path)
    # result_list = doHaar(image_arr)
    result_list = doHog(image_arr)
    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    bbox_arr = detect_faces(input_path)
    image_embeddings = convert_to_embeddings(input_path,bbox_arr)
    results = []
    for i in image_embeddings:
        ce = list(image_embeddings[i][0])
        obj = {
            'iname':i,
            'emb':ce
        }
        results.append(obj)
    data = results
    fe = []
    for i in data:
        img_emb = np.asarray(i['emb'])
        fe.append(img_emb)
    fe = np.asarray(fe)
    
    kmeans_cluster = KMeans(n_clusters=int(K), random_state=0).fit(fe)
    grp = kmeans_cluster.labels_
    cluster = {}
    for i,k in enumerate(data):
        ke = grp[i]
        img_name = k['iname']
        if ke not in cluster:
            cluster[ke] = [img_name]
        else:
            cluster[ke].append(img_name)
    for i in range(int(K)):
        obj = {
            "cluster_no":i,
            "elements":cluster[i]
        }
        result_list.append(obj)
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""



def get_images(input_path):
    image_dict = {}
    for filename in os.listdir(input_path):
        single_img = cv2.imread(os.path.join(input_path,filename))
        if single_img is not None:
            image_dict[filename] = single_img
    return image_dict

def convert_to_embeddings(input_path,bbox_arr):
    img_dict = get_images(input_path)
    fe_dict = {}
    for i in bbox_arr:
        bb_val = i['bbox']
        top,left,bottom,right = bb_val[1],bb_val[0],bb_val[1] + bb_val[3],bb_val[0] + bb_val[2]
        bbox_tuple = [(int(top),int(right),int(bottom),int(left))]
        current_img_name = i['iname']
        single_img = img_dict[current_img_name]
        fe = face_recognition.face_encodings(single_img,bbox_tuple)
        fe_dict[i['iname']] = fe
    return fe_dict


def doHaar(image_arr):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    result = []
    for i in image_arr:
        img = image_arr[i]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_array = face_cascade.detectMultiScale(gray, 1.15, 6)
        for (x,y,w,h) in face_array:
            sx = x -5
            if sx < 0:
                sx =0
            sy = y - 5
            if sy < 0:
                sy = 0
            d = {
                "iname": i,
                "bbox": [float(sx),float(sy),float(w+10),float(h+10)]
            }
            result.append(d)

    return result

def doHog(image_arr):
    faceDetector = dlib.get_frontal_face_detector()
    result = []
    for i in image_arr:
        img = image_arr[i]

        faceRects = faceDetector(img, 1)
        for face in faceRects:
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            x = x - 5
            if x < 0:
                x = 0
            y = y -5
            if y < 0:
                y= 0
            d = {
                "iname": i,
                "bbox": [float(x),float(y),float(w+8),float(h+8)]
            }
            result.append(d)
    return result


