# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Sun Dec 20 14:44:23 2020                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""


import cv2 
import numpy as np
import os 
import matplotlib.pyplot as plt
from matplotlib import transforms




width = 960
height = 480

lower_bound_height = int((height / 4)*2)
higher_bound_height = int((height / 4)*3)


lower_bound_width = int(width / 3)
higher_bound_width = int((width / 3)*2)


#higher_bound = 100
#Mid_screen_width = Mid_screen_height = 112

kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
#roi_corners_rectangle = np.array([[(0,height), (0,higher_bound), (width,higher_bound), (width,height)]], dtype=np.int32)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
white = (255, 255, 255)

#y = np.linspace(0,higher_bound_height-lower_bound_height,higher_bound_height-lower_bound_height+1)
y = np.linspace(0,height,height+1)
		
	
folder = "./output_segmentation_and_extraction/roads"
images = os.listdir(folder)
means = []
medians = []
averages = []


#list_images = ['./dataset/speedbump/tunisia/IMG_0740.JPG','./dataset/speedbump/tunisia/IMG_0743.JPG']
for img in images:
	#print (img)
	means = []
	medians = []
	averages = []
	
	
	fig, ax = plt.subplots()
	#image = cv2.imread(img)
	image = plt.imread(os.path.join(folder, img))
	
	#ax.imshow(image)
	
	#image = cv2.imread(img)
	image = cv2.resize(image,(width,height))
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#image= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	#image, a, b = cv2.split(image)
	
	#mask_1 = np.zeros(image.shape, dtype=np.uint8)
	#cv2.fillPoly(mask_1, roi_corners_rectangle, white)
	#image = cv2.bitwise_and(image, mask_1)
	
	
	#mask_2 = np.zeros(image.shape, dtype=np.uint8)
	#roi_corners = np.array([[(0,height), (Mid_screen_width,Mid_screen_height), (width,height)]], dtype=np.int32)
	#cv2.fillPoly(mask_2, roi_corners, white)
	# apply the mask
	#extracted_road = cv2.bitwise_and(image, mask_2)
	#cv2.imshow("road",extracted_road)
	
	#image = cv2.filter2D(extracted_road, -1, kernel_sharpen)
	#tr = transforms.Affine2D().rotate_deg(90)
	
	
	
	#ROI = image[lower_bound_height:higher_bound_height, lower_bound_width:higher_bound_width]
	implot = plt.imshow(image)
	cv2.imshow('%s'%img,image)
	
	
	#image = clahe.apply(image)
	#image = cv2.merge((image,a,b))
	#image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
	
	#cv2.imshow('%s'%img,image)
	
	for i in range (height):
		line = np.array(image [i,:])
		line = line[line != 0]
		means.append(np.mean(line))
		medians.append(np.median(line))
		averages.append(np.average(line))
		#print (line)
		#plt.plot(line)
		#cv2.imshow('test',line)
		#cv2.waitKey(0)
		#plt.show()
		
		
	means.append(0)	
	medians.append(0)
	averages.append(0)
	#ax.imshow(image)
	
	#ax.invert_yaxis()
	#ax.invert_xaxis()
	ax.plot(means,y,'r--', linewidth=1, label='means')
	ax.plot(medians,y,'b--', linewidth=1, label='medians')
	#ax.plot(averages,y,'g--', linewidth=1, label='averages')
	ax.legend()
	fig.savefig('%s.png'%img)
	#plt.clf()
		#input()
#cv2.destroyAllWindows()
	
	