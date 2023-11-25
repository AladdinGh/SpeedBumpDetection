# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Sun Mar 29 12:44:40 2020                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""
import numpy as np
import cv2
#import pywt


lower_bound = 500
higher_bound = 320
width = 960
height = 640
Mid_screen_width = 480
Mid_screen_height = 320

list_images = ['IMG_1038.jpg','IMG_1240.jpg','IMG_1733.jpg','IMG_1734.jpg','IMG_1964.jpg']

for img_name in list_images:
    
################### read and resize image ################################################################################################
    img = cv2.imread(img_name)
    img = cv2.resize(img,(width,height))
    
################# opening and gray conversion ################################################################################################
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale 
    #cv2.imshow("masked_image",gray)
    
################## delete the unnecessary upper part of the image  ################################################################################################
    mask_1 = np.zeros(gray.shape, dtype=np.uint8)
    roi_corners_rectangle = np.array([[(0,height), (0,higher_bound), (width,higher_bound), (width,height)]], dtype=np.int32)
    white = (255, 255, 255)
    cv2.fillPoly(mask_1, roi_corners_rectangle, white)
    # apply the mask
    masked_image = cv2.bitwise_and(gray, mask_1)
    #cv2.imshow("masked_image",masked_image)
    
################# extract the road from vanishing point  ################################################################################################
    mask_2 = np.zeros(masked_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(0,height), (Mid_screen_width,Mid_screen_height), (width,height)]], dtype=np.int32)
    cv2.fillPoly(mask_2, roi_corners, white)
    # apply the mask
    extracted_road = cv2.bitwise_and(masked_image, mask_2)
    #cv2.imshow("road",extracted_road)
################# extract the template  ###############################################################################################
	#offset = 50
	#height_template = 100
	#mask_3 = np.zeros(extracted_road.shape, dtype=np.uint8)
	#roi_corners_template = np.array([[(0,height-offset), (0,height-offset-height_template), (width,height-offset-height_template), (width,height-offset)]], dtype=np.int32)
	#cv2.fillPoly(mask_3, roi_corners_template, white)
	# apply the mask
	#road = cv2.bitwise_and(extracted_road, mask_3)
	#template_init = road[height-offset-height_template:height-offset, 0:width]
	#cv2.imshow("template",road)
################# Construct haar features  ######################################################################
	
	# idea is to construct one features (with different sizes) train them adaboost
	# labeling can be done so, the program shows me the images I type 0 or 1 , name of file is saved with label
    
    offset_width = 100
    offset_height = 50
    mask_4 = np.zeros(extracted_road.shape, dtype=np.uint8)
    roi_haar_template = np.array([[(Mid_screen_width-offset_width,Mid_screen_height+(2*offset_height)), (Mid_screen_width-offset_width,Mid_screen_height+offset_height), (Mid_screen_width+offset_width,Mid_screen_height+offset_height), (Mid_screen_width+offset_width,Mid_screen_height+(2*offset_height))]], dtype=np.int32)
    cv2.fillPoly(mask_4, roi_haar_template, white)
    # apply the mask
    haar = cv2.bitwise_and(extracted_road, mask_4)
    #cv2.imshow("haar",haar)
    cropped_img = haar[Mid_screen_height+offset_height:Mid_screen_height+(2*offset_height),Mid_screen_width-offset_width:Mid_screen_width+offset_width]
    #cv2.imshow("cropped_img",cropped_img)
    
    # so now we have a cropped image of the road, we will train a classifier with this patch
    h, w = cropped_img.shape
    #print (w,h)
    

################# iterate over the image patches  ###############################################################################################
#	end_index = height-offset-height_template
#	start_index = 1
#	while (start_index > 0):
#		#end_index = height-offset-height_template
#		start_index = end_index-height_template
#		print (start_index,end_index)
#		template = temp[start_index:end_index, 0:width]
#		#cv2.imshow("temp",temp)	
#		cv2.imshow("template",template)	
#		cv2.waitKey(0)
#		
#		############## this is to be run after execution
#		end_index = start_index	
	
################# hoare transform ???????? Can be a good method to be further tested ####################################################
#	titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
#	import matplotlib.pyplot as plt	
#	coeffs2 = pywt.dwt2(extracted_road, 'haar')
#	LL, (LH, HL, HH) = coeffs2
#	#print (coeffs2)
#	fig = plt.figure(figsize=(1000, 3))
#	for i, a in enumerate([LL, LH, HL, HH]):
#		ax = fig.add_subplot(1, 4, i + 1)
#		ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#		ax.set_title(titles[i], fontsize=10)
#		ax.set_xticks([])
#		ax.set_yticks([])
#	plt.show()
#####################  get the structuring element ###########################
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
     



################ save and display the image  ################################################################################################
    outfile = '%s_patch.jpg' % (img_name)
    cv2.imwrite(outfile, cropped_img)
	#cv2.imshow("extracted_road",extracted_road)
	#cv2.imshow("template_init",template_init)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()