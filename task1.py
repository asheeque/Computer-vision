"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters)
    labelled_img,ccd = detection(test_img)
    res = recognition(test_img,ccd)
    return res
    # raise NotImplementedError

def enrollment(images):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
        
        
    des_d = {}
    for im in images:
        name = im[0]
        img = im[1]
        img = np.pad(img, [(2, ), (2, )], mode='constant',constant_values=(255))
        img=cv2.resize(img, (150, 150))
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        if len(kp) < 1:
            continue
        des_d[name]=des.tolist()
    with open("features.json", "w") as outfile:  
    	json.dump(des_d, outfile)
    # raise NotImplementedError

def detection(image):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
   
    binary_img = np.zeros((image.shape)) 
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < 110:
                binary_img[i][j] = 1
    labelled_img, ccd = get_connected_components(binary_img)
    return labelled_img, ccd
     

def get_connected_components(image):

    seq = 1
    ccd = {}
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j]== 1:
                min_x,max_x,width,heigth = bfs(image,i,j,seq,0)
                if seq>0 and width > 1 and heigth>1:
                    ccd[seq] = min_x,max_x,width,heigth
                seq+=1


    return image,ccd



def bfs(image,i,j,seq,level):
    queue = []
    queue.append([i,j])
    dir = [[1,0],[0,1],[1,1],[0,-1],[-1,0],[-1,-1],[1,-1],[-1,1]]
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    while (len(queue)!=0):
        size = len(queue)
        level += 1
        for i in range(size):
            temp = queue.pop()
            for num in range(len(dir)):
                row = dir[num][0]+temp[0]
                col= dir[num][1]+temp[1]
                if(checker(image,row,col,seq)):
                    image[row][col] = seq
                    queue.append([row,col])
                    min_x = min(min_x,col)
                    min_y = min(min_y,row)
                    max_x = max(max_x,col)
                    max_y = max(max_y,row)
                  
    height = max_y - min_y
    width = max_x - min_x
    return [min_x,min_y,width,height]
            
            


def checker (image,i,j,seq):

    if i<0 or j<0 or i>=len(image) or j>=len(image[0]) or image[i][j]==0 or image[i][j]==seq:
        return False
    return True


def recognition(image,ccd):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    sift = cv2.SIFT_create()
    with open('features.json') as file:
        features = json.load(file)

    temp=float('inf')
    d = {}
    for single_feature in features:
        if len(features[single_feature]) == 0:
                continue
        desc2 = np.asarray(features[single_feature])
        if desc2 is None:
            continue

        temp = float('inf')
        for key in ccd:
            
            current_bbox = ccd[key]
            bbox_image = image[current_bbox[1]:current_bbox[1]+current_bbox[3],current_bbox[0]:current_bbox[0]+current_bbox[2]]
            bbox_image =np.pad(bbox_image, [(4, ), (4, )], mode='constant',constant_values=(255))
            bbox_image =cv2.resize(bbox_image , (150, 150))
            kp, des = sift.detectAndCompute(bbox_image,None)
            if des is None:
                continue
            for i in range(len(des)):

                for j in range(len(desc2)):
                    s = np.sum(np.square(np.subtract(des[i],desc2[j])))/128
                    temp=min(temp,s)
                    d[single_feature] = temp

    res = []
    for key in ccd:
        flag = False
        current_bbox = ccd[key]        
        bbox_image = image[current_bbox[1]:current_bbox[1]+current_bbox[3],current_bbox[0]:current_bbox[0]+current_bbox[2]]
        bbox_image =np.pad(bbox_image, [(4, ), (4, )], mode='constant',constant_values=(255))
        bbox_image =cv2.resize(bbox_image , (150, 150))
        kp, des = sift.detectAndCompute(bbox_image,None)
        if des is None:
            x = {
                "bbox":current_bbox,
                "name":"UNKNOWN"
                }
            res.append(x)
            continue
        for single_feature in features:
            if len(features[single_feature]) == 0:
                continue
            desc2 = np.asarray(features[single_feature])
            if desc2 is None:
                continue
            for i in range(len(des)):
                for j in range(len(desc2)):
                    s = np.sum(np.square(np.subtract(des[i],desc2[j])))/128
                    temp=min(temp,s)
                    if s < d[single_feature] * 1.85:
                        x = {
                            "bbox":current_bbox,
                            "name":single_feature
                        }
                        flag = True
                        res.append(x)
            if flag == True:

                flag = False
                continue
        if flag == False:
            x = {
                "bbox":current_bbox,
                "name":"UNKNOWN"
                }
            res.append(x)

    return res
    # raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])
        
    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)
    print(results)
    with open("results.json", "w") as outfile:  
        json.dump(results, outfile)
    # save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
