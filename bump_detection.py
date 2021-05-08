import cv2
import cvlib as cv


import time
from imutils.video import VideoStream
import numpy as np
import matplotlib.pyplot as plt




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
height_image = 512
Epsilon = 0.001
white = (255, 255, 255)
average = -1
cnt=1
name = 0
std_array = []

backSub = cv2.createBackgroundSubtractorMOG2()
print("[INFO] starting video stream...")
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("./videos/2.mp4")
#vs = cv2.VideoCapture("video.avi")
#while(vs.isOpened()):
while(True):
#img = cv2.imread("12.jpg")

    ret,image = vs.read()
    
    image = cv2.resize(image,(width_image,height_image))
    ori = image
    
    
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel2 = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    kernel3 = np.array([[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel2)    
    
    #outfile4= 'output_debugguing/%s_filtered.png' % (name)
    #cv2.imwrite(outfile4,image) 
    
    fig, ax = plt.subplots()
    #image = cv2.resize(image,(width_image,height_image))          
    #boxes, labels, _conf = cv.detect_common_objects(image, model="yolov3-tiny")
    boxes, labels, _conf = cv.detect_common_objects(image, model="yolov3")
    #print(labels, boxes)
    medians = []
    #mask_detected_objects = np.zeros(image.shape, dtype=np.uint8)
    
    
    for i in range(len(boxes)):
        box = boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        #rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        cv2.rectangle(image, (x1,y1), (x2,y2+20), np.NaN, -1)
        
        
        
        
        
    #
    
        
        #crop_img = image[width, height]
        
        
        #image = image - crop_img
        
        ####y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        
        #points = np.array([[910, 641], [206, 632], [696, 488], [458, 485]])
        
        #cv2.polylines(img, [points], 1, (255,255,255))
        ####roi_corners_rectangle = np.array([[x1,y1],
        ####                                   [x1,y2],
        ####                                   [x2,y2],
        ####                                   [x1,y2]]
        ####                                  , dtype=np.int32)
    
        ####cv2.fillPoly(mask_detected_objects, [roi_corners_rectangle], white)
        # apply the mask
        ####image = cv2.bitwise_and(image, mask_detected_objects)
        
        
    #cv2.imshow('video',image)    
        
    
    #################         remove shadows      ################################################################################################
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #fgMask = backSub.apply(image)
    #image = image-fgMask
   
    
    
    #rgb_planes = cv2.split(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_planes = []
    result_norm_planes = []
    #for plane in rgb_planes:
    dilated_img = cv2.dilate(image, np.ones((5,5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 5)
    diff_img = 255 - cv2.absdiff(image, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
    
    image = cv2.merge(result_norm_planes)
    
    #image = cv2.filter2D(image, -1, kernel2) 
    
    outfile3 = 'output_debugguing/%s_removed.png' % (name)
    cv2.imwrite(outfile3,image)    

    
    
    
    ################# extract the road from vanishing point and remoe sidewal and curbstone  ################################################################################################
    offset = (height_image/5)*4
    Mid_screen_width = int(width_image/2)
    Mid_screen_height = int(height_image/2)
    
    mask_2 = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[(0,height_image),(0,offset), (Mid_screen_width,Mid_screen_height),(width_image,offset),(width_image,height_image)]], dtype=np.int32)
    cv2.fillPoly(mask_2, roi_corners, white)
    # apply the mask
    image = cv2.bitwise_and(image, mask_2)
    
   
    ################# cut the upper half   ################################################################################################

    upper_bound = int((height_image/4)*2)
    #upper_bound = int(height_image/2)
    image = image[upper_bound:height_image,:]
    
    
    #image = cv2.resize(image,(width_image,height_image))
    ################# sharpen the edges   ################################################################################################
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # kernel2 = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    # image = cv2.filter2D(image, -1, kernel)
    
    #outfile3 = 'output_debugguing/%s_sharpened.png' % (name)
    #cv2.imwrite(outfile3,image)
    
    
    ###################### compute the median of line pixels only if more than half are non zero pixels
    for i in range (int(image.shape[0])):
        # process line by line
        line = np.array(image [i,:])
        #count the non black pixels 
        #nb_nonblack_pixels = np.count_nonzero(line)
        # process the line only if more than 50% its pixels are not black
        #print(nb_nonblack_pixels)
        #if (nb_nonblack_pixels > int(width_image/10)) :
            # remove the black pixels, we need them not to compute the median
            
           
        #non_zeros = line[line > 0]    
        #black_pixels = line[line == 0]
        #if (len(non_zeros) != 0):
        line = line[line != 0]
            
       
        
    
        #temp = non_zeros[0:len(black_pixels)]
        # if (len(non_zeros) != 0):
        #     np.place(line, line==0, non_zeros)    
        #     m = np.median(line)    
        #     line[line == 0] = m    
   
        # image [i,:] = line
        if(len(line)!=0):
            medians.append(np.median(line))
        
            
   

        
        #print(medians)    
              
        #if(i == 0):
        #    input()
        
        #else :
        #    medians.append(0)
    #print('here is fine')
    # to remove with video input    
    #medians.append(0)
    
    medians = np.array(medians)
   #medians[np.isnan(medians)] = 0
    #print (medians)
    # detect the edges to delete jittering of median vector
    # a scenario for this is , road (median values), no road (0) and then back to road(median values)
    #ind_rising_edge = detect_raising_edge(medians)
    #ind_falling_edge = detect_falling_edge(medians)
    

    # we need this middle part only
    #middle = medians[ind_rising_edge+1:ind_falling_edge-1]
    middle = medians
    
    #middle = middle[np.logical_not(np.isnan(middle))]
    
    
    y = np.linspace(1,len(middle),len(middle))
    # compute the standard deviation of the median vector to detect the jittering
    # in the median vector
    buffer = 5
    if cnt < buffer:
        if (len(middle)!=0):
            std_array.append(np.std(middle))
            #print("std ", np.std(middle))
            cnt = cnt + 1
            print(cnt)
    
   # if there are X frames then we compute the average of std dev array
    
   
    if cnt == buffer:
       #average = np.min(std_array)
       average = np.mean(std_array)
       #average = np.max(std_array)
       print ("##################    the new average is ############  ",average)
       cnt = -1
       std_array = []
            
       
    
         
    # the average has been computed , we can predict
    label = ""
    if (average != -1):       
        label = "speed_bump" if  np.std(middle) - average > Epsilon else "No speed_bump"        
        print ("average is ", average)
        print("std dev of lane is ", np.std(middle))
        print (label)
        
        color = (0, 255, 0) if label == "speed_bump" else (0, 0, 255)
            # display the label and bounding box rectangle on the output
            # frame
        #cv2.putText(image, label, (250, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)   
        #cv2.putText(image, str(np.std(middle)), (240, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, color,1)
        #cv2.putText(image, str(average), (230, 400),cv2.FONT_HERSHEY_SIMPLEX,1, color, 1)
        # save the image for debugging
    ####image = cv2.resize(image,(width_image,height_image))            
    #cv2.imshow('video',image)
        ###name = name + 1
        ###outfile = 'output_debugguing/%s_prediction.jpg' % (name)
        ###cv2.imwrite(outfile,image) 
        
  
    ax.imshow(image)
    #ax.plot(means, y, label='means')
    ax.plot(middle, y, 'r',label='median')
    ax.text(0, 20, label, fontsize=15,color ="green")
    ax.text(0, 50, str(np.std(middle)), fontsize=15,color ="green")
    ax.text(0, 80, str(average), fontsize=15,color ="green")
    #ax.legend()
    name = name + 1
    outfile = 'output_debugguing/%s_test.png' % (name)
    outfile2 = 'output_debugguing/%s_ori.png' % (name)
    cv2.imwrite(outfile2,ori)
    plt.savefig(outfile)
   
    
#plt.show()
#plt.plot(std_array)        
    
    #key = cv2.waitKey(1) & 0xFF
    ## if the `q` key was pressed, break from the loop
    #if key == ord("q"):
    #    break
    #    vs.stop()


   