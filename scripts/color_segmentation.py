# -*- coding: utf-8 -*-
"""

#               Altran Deutschland S.A.S & Co. KG
     
#   \authors   ala.edddine.gharbi@altran.com                                                                             
#   \version   01.00.00                                                                                                  
#   \date      Sat Nov 21 14:49:34 2020                                                                                                
#   \copyright Copyright (c) 2018, Altran Deutschland GmbH. All rights reserved!                                          
#              All rights exclusively reserved for Altran Deutschland S.A.S & Co. KG, unless expressly otherwise agreed.                       
#              Redistribution in source or any other form, with or without modification, is not permitted.               
#              You may use this code under the according license terms of Altran Deutschland S.A.S & Co. KG.                                        
#              Please contact Altran Deutschland S.A.S & Co. KG at ala.edddine.gharbi@altran.com to get the appropriate terms and conditions.                 
                                                                                                           

"""

import numpy as np
from sklearn.cluster import KMeans
import argparse
import cv2
import datetime


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to image file')
ap.add_argument('-w', '--width', type=int, default=0,help='Width to resize image to in pixels')
ap.add_argument('-s', '--color-space', type=str, default='bgr',help='Color space to use: BGR (default), HSV, Lab, YCrCb (YCC)')
ap.add_argument('-c', '--channels', type=str, default='all',help='Channel indices to use for clustering, where 0 is the first channel,'+ ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" '+ 'selects channels B and R. (default "all")')
ap.add_argument('-n', '--num-clusters', type=int, default=3,help='Number of clusters for K-means clustering (default 3, min 2).')
ap.add_argument('-o', '--output-file', action='store_true',help='Save output image (side-by-side comparison of original image and'+ ' clustering result) to disk.')
ap.add_argument('-f', '--output-format', type=str, default='png',help='File extension for output image (default png)')



args = vars(ap.parse_args())
image = cv2.imread(args['image'])

# Resize image and make a copy of the original (resized) image.
if args['width'] > 0:
    height = int((args['width'] / image.shape[1]) * image.shape[0])
    image = cv2.resize(image, (args['width'], height),
        interpolation=cv2.INTER_AREA)
orig = image.copy()

# Change image color space, if necessary.
colorSpace = args['color_space'].lower()
if colorSpace == 'hsv':
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
elif colorSpace == 'lab':
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
else:
    colorSpace = 'bgr'  # set for file naming purposes
	 
	 
# Keep only the selected channels for K-means clustering.
if args['channels'] != 'all':
    channels = cv2.split(image)
    channelIndices = []
    for char in args['channels']:
        channelIndices.append(int(char))
    image = image[:,:,channelIndices]
    if len(image.shape) == 2:
        image.reshape(image.shape[0], image.shape[1], 1)

# Flatten the 2D image array into an MxN feature vector, where M is
# the number of pixels and N is the dimension (number of channels).
reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

# Perform K-means clustering.
if args['num_clusters'] < 2:
    print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
numClusters = max(2, args['num_clusters'])
kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)



# Reshape result back into a 2D array, where each element represents the
# corresponding pixel's cluster index (0 to K - 1).
clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
    (image.shape[0], image.shape[1]))

# Sort the cluster labels in order of the frequency with which they occur.
sortedLabels = sorted([n for n in range(numClusters)],
    key=lambda x: -np.sum(clustering == x))

# Initialize K-means grayscale image; set pixel colors based on clustering.
kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
for i, label in enumerate(sortedLabels):
    kmeansImage[clustering == label] = int((255) / (numClusters - 1)) * i



# Concatenate original image and K-means image, separated by a gray strip.
concatImage = np.concatenate((orig,
    193 * np.ones((orig.shape[0], int(0.0625 * orig.shape[1]), 3), dtype=np.uint8),
    cv2.cvtColor(kmeansImage, cv2.COLOR_GRAY2BGR)), axis=1)
cv2.imshow('Original vs clustered', concatImage)



if args['output_file']:
    # Construct timestamped output filename and write image to disk.
    dt = datetime.datetime.now()
    fileExtension = args['output_format']
    filename = (str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour)
        + str(dt.minute) + str(dt.second) + colorSpace + '_c' + args['channels']
        + 'n' + str(numClusters) + '.' + fileExtension)
    cv2.imwrite(filename, concatImage)
cv2.waitKey(0)













