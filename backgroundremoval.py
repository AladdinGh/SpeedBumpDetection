
import cv2
import cvlib as cv


import time
from imutils.video import VideoStream
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


width_image = 960
height_image = 512

backSub = cv.createBackgroundSubtractorMOG2()

print("[INFO] starting video stream...")
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("./videos/2.mp4")
#while(vs.isOpened()):
while(True):
#img = cv2.imread("12.jpg")

    ret,image = vs.read()
    
    image = cv2.resize(image,(width_image,height_image))
    
    fgMask = backSub.apply(image)
    
    
    cv.rectangle(image, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(image, str(vs.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               #cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv.imshow('Frame', image-fgMask)
    #cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break