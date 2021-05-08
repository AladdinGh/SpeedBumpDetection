# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Wed Dec 23 12:01:47 2020                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""

# To get pi
#import math

# To do linear algebra
#import numpy as np

# To store data
#import pandas as pd

# To create nice plots
#import seaborn as sns

# To count things
#from collections import Counter

# To create interactive plots
#import plotly.graph_objs as go
#from plotly.offline import iplot

# To handle tensors
import tensorflow as tf
#import tensorflow.keras.backend as K

# To handle datasets
#from kaggle_datasets import KaggleDatasets

# To create plots
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap




import cv2
import numpy as np
import os 

width_image = 960
height_image = 480

#folder = "./dataset/no_speed"
folder = "./dataset/mixed_data_set"
#folder = "./dataset/speedbump/germany"
images = os.listdir(folder)
#kernel = np.ones((5,5),np.uint8)


#list_images = ['./dataset/speedbump/2578_prediction4.JPG']

def detect_raising_edge (medians_vector):
    old_value = medians_vector[0]
    #flag_raising = False
    for i in range(len(medians_vector)-1):
        if medians_vector[i] != old_value :
            #flag_raising = True
            break ; 
    return(i)        
    
def detect_falling_edge (medians_vector):
    
    
    #old_value = medians_vector[0]
    #flag_raising = False
    ind = detect_raising_edge(medians_vector)
    if (ind != 0):
        for i in range(ind , len(medians_vector)-1):
            if medians_vector[i]== 255 :
                #print (i)
                #flag_raising = True
                break ; 
        return(i)
    return(0)        


import cvlib as cv


import time
from imutils.video import VideoStream
import numpy as np

name = 0
cnt=0
std_array = []
print("[INFO] starting video stream...")
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("./videos/2.mp4")
#while(vs.isOpened()):
while(True):
#img = cv2.imread("12.jpg")
    ret,image = vs.read()

    cnt = cnt + 1
    #image = plt.imread(os.path.join(folder, img))
    fig, ax = plt.subplots()
    #image = cv2.imread(img)
    
    
    
    #means = []
    medians = []
    averages = []
    
    
    #image_ori = image 
    #image_ori = cv2.resize(image_ori,(512,512))
    #image =  cv2.GaussianBlur(image,(5,5),0)
    
    #width = image.shape[1]
    #height = image.shape[0]
    #lower_bound_height = int((height / 4)*2)
    #higher_bound_height = height
    
    #ROI = image[lower_bound_height:higher_bound_height, :]
    image = cv2.resize(image,(512,512))
    
    #cv2.imshow('blurred',image)
    #cv2.imshow('original',image_ori)
    
    #outfile = 'output_prepare_model/opening/%s_shadowremoved.jpg' % (img)
    #cv2.imwrite(outfile,image)
    
    #prediction_r = model.predict(np.expand_dims(image, axis=0))[0]
    #prediction = tf.argmax(prediction_r, axis=2)
   # Populate the subplots
    #ax1.imshow(image)
    #ax1.set_title('Image: {}'.format(0))
    #ax1.axis('off')
    
    #ax2.imshow(prediction, cmap=cm, norm=norm)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   #prediction = prediction.numpy().flatten()
    
    #prediction= prediction.reshape(512, 512)
    #prediction = np.uint8(prediction)
    #dilation = cv2.dilate(prediction,kernel,iterations = 1)

    #dilation = dilation.flatten()
    #image = image.flatten()    
    #res = image
    #res = np.array([image[i] if (np.abs(prediction[i]-7)<=1)  else 0 for i in range(len(prediction))])
    #res = np.array([image[i] if prediction[i] == 7  else 0 for i in range(len(prediction))])
    #for i in range (image.shape[0]):
    #    for j in range (image.shape[1]):
    #        if prediction[i+j] != 7 :
    #            res[i][j] = 0
    
    
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
    
    ###################### delete the upper half screen, assumption is the road is on the lower half
    crop_img = image[int((height_image/3)*2):height_image,:]
    #cv2.imshow('test',res)
    #implot = plt.imshow(image)
    # MORPH_RECT  MORPH_ELLIPSE   MORPH_CROSS
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3));
    #crop_img = cv2.morphologyEx(crop_im g,cv2.MORPH_OPEN, kernel);
    #erosion = cv2.erode(crop_img,kernel,iterations = 1)
    #opening = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)
    
    ###################### 
    
    upper_bound = int((height_image/3)*2)
    image = image[upper_bound:height_image,:]
    
    
    ###################### compute the median of line pixels only if more than half are non zero pixels
    for i in range (int(image.shape[0])):
        # process line by line
        line = np.array(image [i,:])
        #count the non black pixels 
        nb_nonblack_pixels = np.count_nonzero(line)
        # process the line only if more than 50% its pixels are not black
        if nb_nonblack_pixels > int(width_image/2):
            # remove the black pixels, we need them not to compute the median
            line = line[line != 0]
            medians.append(np.median(line))
        #else :
        #    medians.append(0)
        
    # to remove with video input    
    #medians.append(0)
    medians = np.array(medians)
    #print (medians)
    # detect the edges to delete jittering of median vector
    # a scenario for this is , road (median values), no road (0) and then back to road(median values)
    #ind_rising_edge = detect_raising_edge(medians)
    #ind_falling_edge = detect_falling_edge(medians)
    

    # we need this middle part only
    #middle = medians[ind_rising_edge+1:ind_falling_edge-1]
    middle = medians
    
    # compute the standard deviation of the median vector to detect the jittering
    # in the median vector
    if cnt < 100:
        if (len(middle)!=0):
            std_array.append(np.std(middle))
            #print("std ", np.std(middle))
            cnt = cnt + 1
    
   # if there are 10 frames then we compute the average of std dev array

    if cnt == 4:
       average = np.mean(std_array)
       print ("##################    the new average is ############  ",average)
       cnt = -1       
   
    
    
    y = np.linspace(1,len(middle),len(middle))
    if (len(middle)!=0):
        std_array.append(np.std(middle))
        print("std ", np.std(middle))
        
        #print("Max : ", np.max(middle))
        #print((np.min(middle)+np.max(middle)/2))
    #_ = plt.hist(middle, bins='auto')
    #outfile = 'output_prepare_model/opening/%s_histos.jpg' % (img)
    #plt.savefig(outfile)
    #plt.cla()

#plt.show()
    #averages = np.array(means)
    #needed_points = medians[1:-1]
    ax.imshow(image)
    #ax.plot(means, y, label='means')
    ax.plot(medians, y, 'r',label='median')
    ax.text(2, 6, np.std(middle), fontsize=15)
    ax.legend()
    name = name + 1
    outfile = 'output_prepare_model/%s_test.png' % (name)
    plt.savefig(outfile)
    
plt.show()
plt.plot(std_array)        
    #means = np.interp(means, (means.min(), means.max()), (0, +960))
    #medians = np.interp(medians, (medians.min(), medians.max()), (0, +960))
    
    
    #shallow = np.zeros((width,height,3), dtype=np.int8)
    #data = y,means 
    #shallow = shallow + data 
    #res = image + data 
    
    #ax.invert_yaxis()
    #ax.invert_xaxis()
    #ax.plot(means,y,'r--', linewidth=1, label='means')
    #ax.plot(medians,y,'b--', linewidth=1, label='medians')
    #ax.plot(averages,y,'g--', linewidth=1, label='averages')
    #ax.legend()
    #fig.savefig('output_prepare_model/%s.png'%img)
    
    #ax2.set_title('Prediction: {}'.format(0))
    #ax2.axis('off')
    #outfile = 'output_prepare_model/%s_prediction.jpg' % (img)
    #outfile = '%s_prediction.jpg' % (img)
    #plt.savefig(outfile)
    #cv2.imwrite(outfile,res)
#plt.show()
    

    
    
