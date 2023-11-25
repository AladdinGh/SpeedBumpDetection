# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Sat Nov 21 15:10:12 2020                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""

import numpy as np
from sklearn.cluster import KMeans

import cv2


n_colors = 4
#list_images = ['IMG_0757.jpg_extracted_road.jpg','IMG_1038.jpg_extracted_road.jpg','IMG_1733.jpg_extracted_road.jpg','IMG_1734.jpg_extracted_road.jpg','IMG_1964.jpg_extracted_road.jpg','IMG_1240.jpg_extracted_road.jpg']
list_images = ['IMG_0757.jpg_patch.jpg','IMG_1038.jpg_patch.jpg','IMG_1733.jpg_patch.jpg','IMG_1734.jpg_patch.jpg','IMG_1964.jpg_patch.jpg','IMG_1240.jpg_patch.jpg']

for img in list_images:
	print (img)
	sample_img = cv2.imread(img)
	w,h,_ = sample_img.shape
	sample_img = sample_img.reshape(w*h,3)
	kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(sample_img)
	
	# find out which cluster each pixel belongs to.
	labels = kmeans.predict(sample_img)
	
	# the cluster centroids is our color palette
	identified_palette = np.array(kmeans.cluster_centers_).astype(int)
	
	# recolor the entire image
	recolored_img = np.copy(sample_img)
	for index in range(len(recolored_img)):
	    recolored_img[index] = identified_palette[labels[index]]
	    
	# reshape for display
	recolored_img = recolored_img.reshape(w,h,3)
	outfile = 'kmeans_color_q%s_.jpg' % (img)
	cv2.imwrite(outfile, recolored_img)
