import numpy as np
import cv2


width = 960
height = 640

list_images = ['IMG_1038.jpg','IMG_1240.jpg','IMG_1733.jpg','IMG_1734.jpg','IMG_1964.jpg']
img_name = 'IMG_1038.jpg'
for img_name in list_images:
    
################### read and resize image ################################################################################################
    img = cv2.imread(img_name)
    img = cv2.resize(img,(width,height))
    
################# gray conversion ################################################################################################
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale 

################# smoothing ################################################################################################   
    kernel = np.ones((5,5),np.float32)/25
    smoothed = cv2.filter2D(gray,-1,kernel)
    
    
    
    
    
    
    
    
    
    
    
    
    cv2.imshow("smoothed image",smoothed)
    cv2.waitKey(0) 
cv2.destroyAllWindows()

