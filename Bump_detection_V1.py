
import cv2
import cvlib as cv


import time
from imutils.video import VideoStream
import numpy as np
import matplotlib.pyplot as plt


# to reshape the output of the model
width_image = 960
height_image = 512
Epsilon = 0.001
white = (255, 255, 255)
average = -1
cnt=1
name = 0
std_array = []

kernel2 = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
print("[INFO] starting video stream...")

vs = cv2.VideoCapture("result.avi")
#vs = cv2.VideoCapture("video.avi")

while(True):

# ====================== read frame , resize it , save original for debugging =================================
    ret,image = vs.read()
    image = cv2.resize(image,(width_image,height_image))
    ori = image
# =============================================================================
    

# ============================= sharpen the edges ===================================
    image = cv2.filter2D(image, -1, kernel2)    
# =============================================================================

# ================================== Inits ================================
    fig, ax = plt.subplots()
    medians = []
# =============================================================================
    
    
# ===================== detect the objects from yolo and extract the bouding boxes========================
# to read further aboutyolo and cvlib https://docs.cvlib.net/object_detection/
    boxes, labels, _conf = cv.detect_common_objects(image, model="yolov3")
# =============================================================================




   
# =====================  loop over the bouding boxes and draw them on the image (with black pixels)========================  
    
    for i in range(len(boxes)):
        box = boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        cv2.rectangle(image, (x1,y1), (x2,y2+20), ((0,0,0)), -1)
        
# =====================================================================================================
    
    
    
# ======================== Remove shadows from image (trees ...)========================================
# Transform to grey, dilate and blur and then compute the difference.
# this block is taken as a black box and need to be investigated further
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_planes = []
    result_norm_planes = []
    dilated_img = cv2.dilate(image, np.ones((5,5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 5)
    diff_img = 255 - cv2.absdiff(image, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)    
    image = cv2.merge(result_norm_planes) 
# =============================================================================
  

    
#==================== extract the road from vanishing point and remove sidewalk and curbstone  =====================
 # manually computed: it is assumed that whatever is on the sides of the image is not a road, therefore removed   
    offset = (height_image/5)*4
    Mid_screen_width = int(width_image/2)
    Mid_screen_height = int(height_image/2)
    
    mask_2 = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[(0,height_image),(0,offset), (Mid_screen_width,Mid_screen_height),(width_image,offset),(width_image,height_image)]], dtype=np.int32)
    cv2.fillPoly(mask_2, roi_corners, white)
    # apply the mask
    image = cv2.bitwise_and(image, mask_2)
    
   
#==================== cut the upper part  ====================
# manually done: it is assumed that whatever is above 1/2  of the image is sky and therefore removed  
    upper_bound = int(height_image/2)
    image = image[upper_bound:height_image,:]
    
   
    

#==================== compute the median of line pixels only if more than half are non zero pixels =================
    for i in range (int(image.shape[0])):
        # process line by line
        line = np.array(image [i,:])
        line = line[line != 0]

        if(len(line)!=0):
            medians.append(np.median(line))
           
    medians = np.array(medians)

# ===============================================================================================================
 # now we computed the median value of pixels line by line, we compute the standard deviation of this metric
 # ide is , if there is a strong jittering then a bumb is detected
    y = np.linspace(1,len(medians),len(medians))
    # compute the standard deviation of the median vector to detect the jittering
    # in the median vector
    buffer = 5
    
    # for 5 frames, we compute the mean of the jittering and use the value as a groundtruth
    if cnt < buffer:
         if (len(medians)!=0):
             std_array.append(np.std(medians))
             cnt = cnt + 1
             print(cnt)
    
    # if there are X frames then we compute the average of std dev array
   
    if cnt == buffer:
        average = np.mean(std_array)
        print ("##################    the new average is ############  ",average)
        cnt = -1
        std_array = []
            
 # ===============================================================================================================    
    
         
    # the average has been computed , we can predict
    label = ""
    if (average!=-1):       
        # label = "speed_bump" if  upper_part_average - lower_part_average > Epsilon else "No speed_bump"        
        # print ("average is ", lower_part_average)
        # print("std dev of lane is ", upper_part_average)
        label = "speed_bump" if  np.std(medians) - average > Epsilon else "No speed_bump"        
        print ("average is ", average)
        print("std dev of lane is ", np.std(medians))
        print (label)

# ===================================== images outputs can be deleted for dubugging=================================================
    ax.imshow(image)
    str1 = "avg " + str(average)
    str2 = "current " + str(np.std(medians))
    #ax.plot(means, y, label='means')
    ax.plot(medians, y, 'r',label='median')
    ax.text(0, 100, label, fontsize=15,color ="green")
    ax.text(0, 150, str1, fontsize=15,color ="green")
    ax.text(0, 180, str2, fontsize=15,color ="green")
    #ax.legend()
    name = name + 1
    #print(name)
    outfile = 'output_debugguing/%s_test.png' % (name)
    outfile2 = 'output_debugguing/%s_ori.png' % (name)
    cv2.imwrite(outfile2,image)
    plt.savefig(outfile)
   

   