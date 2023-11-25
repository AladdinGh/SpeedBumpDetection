# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Thu Nov 12 08:00:00 2020                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


'''
def convolve2d(image, kernel):
    """
    This function which takes an image and a kernel and returns the convolution of them.

    :param image: a numpy array of size [image_height, image_width].
    :param kernel: a numpy array of size [kernel_height, kernel_width].
    :return: a numpy array of size [image_height, image_width] (convolution output).
    """
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(image)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 4, image.shape[1] + 4))
    image_padded[2:-2, 2:-2] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x]=(kernel * image_padded[y: y+3, x: x+5]).sum()

    return output
 
'''
lower_bound = 500
higher_bound = 320
width = 960
height = 640
Mid_screen_width = 480
Mid_screen_height = 320



width = 200
height = 120
#'IMG_1038.jpg_patch.jpg','IMG_1733.jpg_patch.jpg','IMG_1734.jpg_patch.jpg','IMG_1964.jpg_patch.jpg'
list_images = ['IMG_0757.jpg_patch.jpg']

for img_name in list_images:
	src = cv2.imread(img_name)
	src = cv2.resize(src,(width,height))
	#src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	#src = cv2.fastNlMeansDenoisingColored(src,None,10,10,7,21)
	#cv2.imshow("src",src)
	'''
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	
	#image_sharpen = convolve2d(gray, kernel=kernel)
	
	#dilation = cv2.dilate(image_sharpen, kernel,5)
	# kernel to be used to get sharpened image
	# rectangular [[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]]
	# cross shaped [[0, 0, 1, 0, 0],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[0, 0, 1, 0, 0]]
	# elliptical [[0, 0, 1, 0, 0],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[0, 0, 1, 0, 0]]
	#KERNEL = np.array([[0, 0, 1, 0, 0],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[0, 0, 1, 0, 0]])
	#image_sharpen = convolve2d(gray, kernel=KERNEL)
	#cv2.imwrite('dilation.jpg', image_sharpen)
	

	#laplacian = cv2.Laplacian(gray,cv2.CV_16S)
	#sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
	#sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
	
	#laplacianx64f = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
	#abs_laplacian64f = np.absolute(laplacianx64f)
	#laplacian_8u = np.uint8(abs_laplacian64f)
	

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5));
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
	#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))
	#kernel = np.float32([[-30000, -30000, -30000],[30000, 30000, 30000],[-30000, -30000, -30000]])
	# MORPH_OPEN, MORPH_CLOSE MORPH_GRADIENT MORPH_DILATE MORPH_TOPHAT MORPH_BLACKHAT
	result = cv2.morphologyEx(src,cv2.MORPH_GRADIENT, kernel);
	cv2.imwrite('1.jpg', result)
	result = cv2.Canny(result,45,150,3)

	laplacianx64f = cv2.Sobel(result,cv2.CV_64F,0,1,ksize=3)
	abs_laplacian64f = np.absolute(laplacianx64f)
	laplacian_8u = np.uint8(abs_laplacian64f)

	cv2.imshow("result",result)

	kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
	#kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])
	src = cv2.filter2D(src, -1, kernel)
	cv2.imshow("edges",src)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3));
	src = cv2.morphologyEx(src,cv2.MORPH_GRADIENT, kernel);
	
	
	
	################## delete the unnecessary upper part of the image  ################################################################################################
	mask_1 = np.zeros(src.shape, dtype=np.uint8)
	roi_corners_rectangle = np.array([[(0,height), (0,higher_bound), (width,higher_bound), (width,height)]], dtype=np.int32)
	white = (255, 255, 255)
	cv2.fillPoly(mask_1, roi_corners_rectangle, white)
    # apply the mask
	masked_image = cv2.bitwise_and(src, mask_1)
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
	'''
	# idea is to construct one features (with different sizes) train them adaboost
	# labeling can be done so, the program shows me the images I type 0 or 1 , name of file is saved with label
	#src = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
	#src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
	r, g, b = cv2.split(src)
	cv2.imshow("extracted_road",src)

################################ visualize the color space and make some segmentation ############
	
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib import colors
	
	fig = plt.figure()
	axis = fig.add_subplot(1, 1, 1, projection="3d")
	
	'''
	pixel_colors = src.reshape((np.shape(src)[0]*np.shape(src)[1], 3))
	norm = colors.Normalize(vmin=-1.,vmax=1.)
	norm.autoscale(pixel_colors)
	pixel_colors = norm(pixel_colors).tolist()
	axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
	axis.set_xlabel("Red")
	axis.set_ylabel("Green")
	axis.set_zlabel("Blue")
	plt.show()
	'''

	
	light_white = (150, 150, 150)
	dark_white = (255, 255, 255)

	'''

	from matplotlib.colors import hsv_to_rgb
	lo_square = np.full((10, 10, 3), light_white, dtype=np.uint8) / 255.0
	do_square = np.full((10, 10, 3), dark_white, dtype=np.uint8) / 255.0
	plt.subplot(1, 2, 1)
	plt.imshow(do_square)
	plt.subplot(1, 2, 2)
	plt.imshow(lo_square)
	plt.show()
	
	'''
	
	mask = cv2.inRange(src, light_white, dark_white)
	result = cv2.bitwise_and(src, src, mask=mask)
	plt.subplot(1, 2, 1)
	plt.imshow(mask, cmap="gray")
	plt.subplot(1, 2, 2)
	plt.imshow(result)
	plt.show()
	
	'''

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
    #h, w = cropped_img.shape
    #print (w,h)
    
	offset = 50 
	image_slice = extracted_road[320:420,0:width]
	cv2.imshow("roi",image_slice)
	
	
	#offset = 1
	#roi = result[height-offset:height,0:width]
	#canny_roi = cv2.Canny(src,1,150,3)
	#roi = cv2.morphologyEx(roi,cv2.MORPH_GRADIENT, kernel);
	
	laplacianx64f = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=5)
	abs_laplacian64f = np.absolute(laplacianx64f)
	laplacian_8u = np.uint8(abs_laplacian64f)
	# iterate through the images slices,compare the pixel intensities 
	#src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	#cv2.imshow("roi",laplacian_8u)
	cv2.waitKey(0)

	for i in range (0,height,offset):
		image_slice = src[i:i+offset,0:width]
		#canny_slice = cv2.Canny(image_slice,45,150,3)
		#image_slice = cv2.morphologyEx(image_slice,cv2.MORPH_GRADIENT, kernel);
		#print ("%d %d"%(i,i+offset))
		#outfile = '%s_%s.jpg' % (i,i+offset)
		print (image_slice.mean())
		#cv2.imshow("canny_slice",canny_slice)
		#cv2.imwrite(outfile,image_slice-roi)
		
'''
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	