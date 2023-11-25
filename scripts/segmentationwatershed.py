# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Thu Nov 26 19:18:27 2020                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt






width = 900
height = 500
#'IMG_1038.jpg_patch.jpg','IMG_1733.jpg_patch.jpg','IMG_1734.jpg_patch.jpg','IMG_1964.jpg_patch.jpg'
list_images = ['IMG_0757.jpg_patch.jpg']

for img_name in list_images:
	img = cv2.imread(img_name)
	img = cv2.resize(img,(width,height))
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
	#th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	cv2.imshow("thresh", thresh)
	
	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	# sure background area
	cv2.imshow("opening", opening)
	
	sure_bg = cv2.dilate(opening,kernel,iterations=3)
	cv2.imshow("sure_bg", sure_bg)
	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,3)
	cv2.imshow("dist_transform", dist_transform)
	
	
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)

	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	
	markers = cv2.watershed(img,markers)
	img[markers == -1] = [255,0,0]
	#cv2.imshow("res", img)
	cv2.waitKey(0)
	