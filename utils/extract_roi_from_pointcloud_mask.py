import cv2
import numpy as np
import glob
import os
import base64
import PIL.Image
import cStringIO
# from io import StringIO
import random
from collections import OrderedDict
from dict2xml import dict2xml

rgb_data_path = "../dataset/rgb"
background_data_path = "../dataset/background"
augmented_data_path = "../dataset/aug_data"
background_imgs = []

if __name__ == "__main__":
    #load background image
    for backgroundfile in glob.glob(os.path.join(background_data_path, '*jpg')):
        background_imgs.append(cv2.imread(backgroundfile))

    target_obj_rgbs = []
    target_obj_masks = []
    target_obj_labels = []
    target_obj_countour_points = []
    target_obj_roi_dimensions = []
    for obj_num in glob.glob(os.path.join(rgb_data_path, '*')):
        for rgbfile in glob.glob(os.path.join(obj_num, '*jpg')):
            suffix =  rgbfile.split("/")[-1]
            size = os.path.getsize(rgbfile)
            maskfile = rgbfile.replace("rgb","mask")
            mask_img = cv2.imread(maskfile)
            gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            img2 = cv2.medianBlur(gray,5)
            ret, thresh = cv2.threshold(img2, 1, 255, 0)
            # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            points = np.empty((0,2),int)            
            for i in range(len(contours)):    
                contour_points =  contours[i].reshape([-1,2])
                points= np.append(points,contour_points,axis=0)
            sampled_point = points[::8]
            sorted_point = np.sort(points,axis=0)
            rgb_img = cv2.imread(rgbfile)
            # min_x = sorted_point[0][0] - (sorted_point[-1][0] - sorted_point[0][0])
            # min_y = sorted_point[0][1] - (sorted_point[-1][1] - sorted_point[0][1])
            # max_x = sorted_point[-1][0] + (sorted_point[-1][0] - sorted_point[0][0])
            # max_y = sorted_point[-1][1] + 10
            ## reduce enlarge aera
            min_x = max(sorted_point[0][0] - (sorted_point[-1][0] - sorted_point[0][0]), sorted_point[0][0] - 50)
            min_y = max(sorted_point[0][1] - (sorted_point[-1][1] - sorted_point[0][1]), sorted_point[0][1] - 25)
            max_x = min(sorted_point[-1][0] + (sorted_point[-1][0] - sorted_point[0][0]), sorted_point[-1][0] + 50)
            max_y = sorted_point[-1][1] + 25
            if(min_x<0):
                min_x = 0
            if(min_y<0):
                min_y = 0
            if(max_x>rgb_img.shape[1]):
                max_x = rgb_img.shape[1]
            if(max_y>rgb_img.shape[0]):
                max_y = rgb_img.shape[0]

            height = max_y - min_y
            width  = max_x - min_x
            roi_dimension  = [min_y,min_x,height,width]
            target_obj = rgb_img[min_y:max_y,min_x:max_x]

            rgbObjectfile = rgbfile.replace("rgb","rgb_object")
            file_length = len(rgbfile.split("/")[-1])
            isExist = os.path.exists(rgbObjectfile[:-file_length])
            if not isExist:
                # Create a new directory because it does not exist 
                os.makedirs(rgbObjectfile[:-file_length])

            cv2.imwrite(rgbObjectfile,target_obj)

