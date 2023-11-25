# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Thu Jun 27 14:38:00 2019                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""

import cv2
import numpy as np

import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


#import imutils
def nothing(x):
	pass

#
#img = cv2.imread('IMG_1038.jpg')
#
#img = cv2.resize(img,(960,640))
#
##cv2.imshow('image',img)
#cv2.namedWindow('image')
#cv2.createTrackbar('parm1','image',0,100,nothing)
#cv2.createTrackbar('parm2','image',0,100,nothing)
#cv2.createTrackbar('parm3','image',0,100,nothing)
#cv2.createTrackbar('parm4','image',0,10,nothing)
#
#cv2.createTrackbar('parm5','image',0,100,nothing)
#cv2.createTrackbar('parm6','image',0,10,nothing)
#cv2.createTrackbar('parm7','image',0,10,nothing)




imgL = cv2.imread('IMG_1038.jpg')
#imgR = cv2.imread('right.png')

imgL = cv2.resize(imgL,(960,640))
#imgR = cv2.resize(imgR,(960,640))

#cap = cv2.VideoCapture(0)

while(1):
	
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
#	
#	parm1 = cv2.getTrackbarPos('parm1','image')
#	parm2 = cv2.getTrackbarPos('parm2','image')
#	parm3 = cv2.getTrackbarPos('parm3','image')
#	parm4 = cv2.getTrackbarPos('parm4','image')
#	parm5 = cv2.getTrackbarPos('parm5','image')
#	parm6 = cv2.getTrackbarPos('parm6','image')
#	parm7 = cv2.getTrackbarPos('parm7','image')
#	
#	#cv2.imshow('image',img)
##	im = cv2.Canny(HSV, parm1, parm2)
##	lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, parm3, parm4);
##	for line in lines[0]:
##	    pt1 = (line[0],line[1])
##	    pt2 = (line[2],line[3])
##	    cv2.line(edges, pt1, pt2, (0,0,255), 3)
##	#cv2.imwrite("./2.png", img)
##	cv2.imshow('image',edges)
##	cv2.waitKey(0)
##	
##	ret,thresh = cv2.threshold(gray,parm5,parm6,parm7)
##	#contours1, _, a = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
##	#image1, contours1, hierarchy1 = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
##	#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##	items = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##	#cnts = imutils.grab_contours(contours)
##	contours = items[0] if len(items) ==2 else items[1]
##	
##	out = cv2.drawContours(img, contours, -1, (0,250,0),1)
##	cv2.imshow('image',out)
#	
#	GRAY = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
##	HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
##	YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
##	YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
#	XYZ = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
##	
##	cv2.imshow('image ',img)
##	cv2.imshow('image HSV',HSV)
##	cv2.imshow('image YUV',YUV)
##	cv2.imshow('image YCrCb',YCrCb)
##	cv2.imshow('image XYZ',XYZ)
#	
##get the luminance channel
##	XYZ[:, :, 0] = 0
##	XYZ[:, :, 2] = 0
### set the 	
##	lower_sat = np.array([0,0,0])
##	upper_sat = np.array([parm1,parm2,parm3])
##	
##	mask_sat = cv2.inRange(XYZ, lower_sat, upper_sat)
##	mask_yw_image = cv2.bitwise_and(XYZ, XYZ, mask=mask_sat)
##	
#	cv2.imshow('image HSV',mask_yw_image)
	
	
	
#	XYZ1 = cv2.cvtColor(imgL, cv2.COLOR_RGB2XYZ)
#	XYZ2 = cv2.cvtColor(imgR, cv2.COLOR_RGB2XYZ)
#	
#	XYZ1[:, :, 0] = 0
#	XYZ1[:, :, 2] = 0
#	
#	XYZ2[:, :, 0] = 0
#	XYZ2[:, :, 2] = 0
#	
#	
#	framel_new=cv2.cvtColor(XYZ1, cv2.COLOR_BGR2GRAY)
#	framer_new=cv2.cvtColor(XYZ2, cv2.COLOR_BGR2GRAY)
#
#
#
#	stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
#	disparity = stereo.compute(framel_new,framer_new)
#
#	norm_image = cv2.normalize(disparity, None, alpha = -parm1, beta = parm2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#	threshold = cv2.threshold(norm_image, parm3, parm4, cv2.THRESH_BINARY)[1]
##
#	kernel = np.ones((12,12),np.uint8)
##	morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
##	
#	cv2.imshow('disparity',threshold)
	#cv2.waitKey()
	#plt.imshow(disparity,'gray')
	#plt.show()
	
	
	
	YUV = cv2.cvtColor(imgL, cv2.COLOR_BGR2YUV)
	YUV[:, :, 2] = 0
	YUV[:, :, 1] = 0
	
	HSV = cv2.cvtColor(imgL, cv2.COLOR_RGB2HSV)
	HSV[:, :, 1] = 0
	HSV[:, :, 2] = 0
	
	
	XYZ = cv2.cvtColor(imgL, cv2.COLOR_RGB2XYZ)
	XYZ[:, :, 0] = 0
	XYZ[:, :, 2] = 0
	
	#YUV+HSV
	
	
	#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	
	#bkgnd = cv2.bgsegm.createBackgroundSubtractorMOG()
	#bkgnd = cv2.bgsegm.BackgroundSubtractorMOG() 
	#bkgnd = cv2.bgsegm.createBackgroundSubtractorMOG()
	#fgbg = cv2.BackgroundSubtractorMOG()
	#trhrfg = cv2.bgsegm.createBackgroundSubtractorMOG()
	
	
	#img = YUV+HSV
	GRAY = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(GRAY,150,150,apertureSize = 3)
	
#	lines = cv2.HoughLinesP(edges, 1,np.pi/180 , parm1,parm2,parm3);
#	for line in lines[0]:
#		pt1 = (line[0],line[1])
#		pt2 = (line[2],line[3])
#		cv2.line(img, pt1, pt2, (0,255,0), 2)
		
#	fgbgmask = fgbg.apply(imgL)
#	cv2.imshow('YUV+HSV',fgbgmask)
#	cv2.imshow('original',imgL)
	
	
	#ret, frame = cap.read()
	#GRAY = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	#fgmask = fgbg.apply(GRAY)
	
#	s = (640,960)
#	np.ones(s)"
	
	#rect =cv2.rectangle(GRAY,(0,0),(0,400),(0, 255, 0))
	
	rect =cv2.rectangle(imgL,(960,450),(0,400),(0, 255, 0))
	#print (rect.shape)
	img = GRAY
	
	g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
	
	filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
	
	cv2.imshow('image', img)
	cv2.imshow('filtered image', filtered_img)
	
#	h, w = g_kernel.shape[:2]
#	g_kernel = cv2.resize(filtered_img, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
#	cv2.imshow('gabor kernel (resized)', g_kernel)
	
#	kernel = np.ones((5,5),np.float32)/25
#	dst = cv2.filter2D(YUV+HSV,-1,kernel)
	#cv2.imshow('test Cam',filtered_img)
	#cv2.imshow('original',frame)
	
cv2.destroyAllWindows()























