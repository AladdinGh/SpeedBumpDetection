# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:17:36 2021

@author: galaeddine
"""

from imutils.video import VideoStream
import cv2


vs = cv2.VideoCapture(0)
#vs = cv2.VideoCapture("video.avi")

while(True):
    
    
    ret, frame = vs.read()
    
    
    cv2.imshow("test",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

vs.release()
cv2.destroyAllWindows()