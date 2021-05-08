#import numpy as np
import cv2
import time


cap = cv2.VideoCapture('videos/3.mp4')
#vs = VideoStream("videos/3.mp4").start()

#fpbg=cv2.createBackgroundSubtractorKNN()

fpbg=cv2.createBackgroundSubtractorMOG2()


#fpbg=cv2.createBackgroundSubtractorGMG()
cnt=0

width = 960
height = 480

import numpy as np
import cv2
from matplotlib import pyplot as plt

while(1):
    ret, frame = cap.read()
    cnt = cnt + 1

    #frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
   # frame_orig = cv2.resize(frame,(width,height))
    #frame_orig = frame[int(height/2):height,:]
   
    
    
    

    
   
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    
    
    fgmask = fpbg.apply(thresh)
    
    
    
    
    #fgmask_invert  = (255-thresh)
  
    extracted_road =  thresh - fgmask
    #extracted_road = cv2.bitwise_and(gray, fgmask)
    
    
    extracted_road = extracted_road[int(frame.shape[0]/2):frame.shape[0],:]
    cv2.imshow('frame',extracted_road)
    
    
    #cv2.imshow('frame',fgmask)
    outfile = 'output_video_detection/%s_prediction.jpg' % (cnt)
    
    cv2.imwrite(outfile,extracted_road)
    
    #time.sleep(1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()