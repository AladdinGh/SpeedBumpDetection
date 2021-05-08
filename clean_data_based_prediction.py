# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:59:12 2021

@author: galaeddine
"""
# -*- coding: utf-8 -*-


# To handle tensors
import tensorflow as tf
#import tensorflow.keras.backend as K
import cvlib as cv

# To create plots
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

from tensorflow import keras

import cv2
import numpy as np

# to remove when using video input
import os 



# RGB colors of the classes
COLORS = [(80/255, 168/255, 250/255),
          (242/255, 130/255, 30/255),
          (50/255, 50/255, 50/255),
          (27/255, 44/255, 129/255),
          (163/255, 68/255, 222/255),
          (115/255, 0/255, 0/255),
          (255/255, 255/255, 255/255),
          (191/255, 191/255, 191/255),
          (150/255, 150/255, 150/255),
          (22/255, 146/255, 0/255),
          (245/255, 239/255, 46/255),
          (181/255, 103/255, 10/255),
          (235/255, 0/255, 0/255)]

# Number of classes
NUM_CLASSES = 13


cm = LinearSegmentedColormap.from_list('semantic_map', COLORS, N=NUM_CLASSES)

# Normalize the labels
norm = Normalize(vmin=0, vmax=12)


def detect_raising_edge (medians_vector):
    
    old_value = medians_vector[0]
    for i in range(len(medians_vector)-1):
        if medians_vector[i] != old_value :
            break ; 
    return(i)        
    
def detect_falling_edge (medians_vector):
    
    ind = detect_raising_edge(medians_vector)
    if (ind != 0):
        for i in range(ind , len(medians_vector)-1):
            if medians_vector[i]== 255 :
                break ; 
        return(i)
    return(0)   


# to reshape the output of the model
width_image = 960
height_image = 480


folder = "./output_video_detection/old"
#folder = "./dataset/mixed_data_set"
images = os.listdir(folder)

#fpbg=cv2.createBackgroundSubtractorMOG2()
fpbg=cv2.createBackgroundSubtractorKNN(1, 10, True)


std_array = []
for img in images:

    image = cv2.imread(os.path.join(folder, img))
    print (img)
    
    # to remove with video input
    fig, ax = plt.subplots()

    ####################### median vector for storing the medians of line pixels
    medians = []
    
    ####################### reshape for model input
    image = cv2.resize(image,(width_image,height_imagecls))
    
    ###################### extract the road
    #prediction_r = model.predict(np.expand_dims(image, axis=0))[0]
    #prediction = tf.argmax(prediction_r, axis=2)
    
    
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #prediction = prediction.numpy().flatten()
    
    #image = image.flatten() 
    
    ####################### result to our road segmentation
    #res = np.array([image[i] if prediction[i] == 7  else 0 for i in range(len(prediction))])

    #res= res.reshape(512,512)
    #res = res.astype(np.uint8)
    
    #image=cv2.resize(res,(width,height))
        
    boxes, labels, _conf = cv.detect_common_objects(image, model="yolov3")
    #print(labels, boxes)
    medians = []
    
    for i in range(len(boxes)):
        box = boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        #rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,0), -1)
   
 
    
    ###################### remove the shadow
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 5)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    image = cv2.merge(result_planes)


    image = cv2.resize(image,(width_image,height_image))
    print (image.shape)
    ###################### compute the median of line pixels only if more than half are non zero pixels
    for i in range (int(height/2)):
        # process line by line
        line = np.array(image [i,:])
        #count the non black pixels 
        nb_nonblack_pixels = np.count_nonzero(line)
        # process the line only if more than 50% its pixels are not black
        if nb_nonblack_pixels > int(width/2):
            # remove the black pixels, we need them not to compute the median
            line = line[line != 0]
            medians.append(np.median(line))
        else :
            medians.append(0)
        
    # to remove with video input    
    medians.append(0)
    medians = np.array(medians)
    
    # detect the edges to delete jittering of median vector
    # a scenario for this is , road (median values), no road (0) and then back to road(median values)
    ind_rising_edge = detect_raising_edge(medians)
    ind_falling_edge = detect_falling_edge(medians)
    
    # delete all values in extremities
    # delete with video input
    start = medians[:ind_rising_edge]
    end = medians[ind_falling_edge+1:] 
    # we need this middle part only
    middle = medians[ind_rising_edge+1:ind_falling_edge-1]
    start = [0] * len(start)
    end = [0] * len(end)

    # compile the whole thing
    medians=np.concatenate((start, middle,end), axis=None)
    y = np.linspace(1,len(medians),len(medians))
    
    # compute the standard deviation of the median vector to detect the jittering
    # in the median vector
    if (len(middle)!=0):
        std_array.append(np.std(middle))
        print("std ", np.std(middle))
    # delete with video input    
    ax.imshow(image)
    ax.plot(medians, y, 'r',label='median')
    ax.text(2, 6, np.std(middle), fontsize=15)
    ax.legend()
    outfile = 'output_prepare_model/opening/%s_test.png' % (img)
    plt.savefig(outfile)
    
    
# to be removed with video input    
plt.show()
# for testing, can be deleted
plt.plot(std_array)        

    

    
    
