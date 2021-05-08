# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:56:37 2021

@author: galaeddine
"""


# To handle tensors
import tensorflow as tf
#import tensorflow.keras.backend as K



from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras

import cv2
import numpy as np


import time


base_model = tf.keras.applications.MobileNetV2(input_shape=[512, 512, 3], include_top=False)
    
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
            # we do aggressive exponential smoothing of batch norm
            # parameters to faster adjust to our new dataset
        layer.momentum = 0.9
                
                
layer_names = [
    'block_1_expand_relu',   
    'block_3_expand_relu',   
    'block_6_expand_relu',   
    'block_13_expand_relu',  
    'out_relu',      
]
m_layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=m_layers)
down_stack.trainable = True
     
     
     
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.3))

    result.add(layers.ReLU())

    return result


def ResidualBlock(filters, size, pad='same'):
    initializer1 = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    initializer2 = tf.random_normal_initializer(0., 0.02)
    
    model = tf.keras.Sequential()
#     model.add(ReflectionPadding2D())
    model.add(tf.keras.layers.Conv2D(filters,size, padding=pad, kernel_initializer=initializer1, use_bias=False))
    model.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    model.add(tf.keras.layers.Activation('relu'))
#     model.add(ReflectionPadding2D())
    model.add(tf.keras.layers.Conv2D(filters,size, padding=pad, kernel_initializer=initializer2, use_bias=False))
    model.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))  
    return model
 
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, x1, x2):
        
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(x1) + self.W3(x2)))


        score = self.V(attention_hidden_layer)

        attention_weights = tf.nn.softmax(score, axis=-1)

        context_vector = attention_weights * features
        
        return context_vector
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'W1': self.W1,
            'W2': self.W2,
            'W3': self.W3,
            'V': self.V
        })
        return config



def unet_model(output_channels=13):
    
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])
    
    x = inputs
    
    at = BahdanauAttention(128)(x, x, x)
    
    x = tf.keras.layers.add([x, at])
    
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    
    f_inputs = x
    
    mid_skips = []
    
    res_block = [
        ResidualBlock(320,3, 'same'),
        ResidualBlock(320,3, 'same'),
        ResidualBlock(320,3, 'same'),
#         ResidualBlock(2048,3, 'same'),
#         ResidualBlock(1024,1, 'same'),
#         ResidualBlock(1024,1, 'same'),
    ]
    
    up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),   # 32x32 -> 64x64
]
    

    
    mid_res = [ [ResidualBlock(64,3, 'same'),ResidualBlock(64,3, 'same')], [ResidualBlock(128,3, 'same'),ResidualBlock(128,3, 'same')], 
              [ResidualBlock(256,3, 'same'),ResidualBlock(256,3, 'same')],
              [ResidualBlock(512,3, 'same'),ResidualBlock(512,3, 'same')], [0] ]    
    
    
  # Downsampling through the model
    skips = down_stack(x)
#     x = skips[-1]

    
    
    for skip, res_down in zip(skips, mid_res):
        x = skip
        x1 = x
        if 0 not in res_down:
            for res in res_down:
                x1 = res(x1)
                at = BahdanauAttention(128)(x1, x1, x1)
                x1 = tf.keras.layers.add([x1, at])
                x1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x1)
                
            mid_skips.append(x1)  
            
    temp = x
    
    
    initializer = tf.random_normal_initializer(0., 0.02)
    
    x = layers.Conv2D(320, 3, padding='same', kernel_initializer=initializer)(x)
    
    x = layers.LeakyReLU()(x)
    
    at = BahdanauAttention(128)(x, x, x)
    
    x = tf.keras.layers.add([x, at])
    
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

    
    for r in res_block:
        x1 = x
        x2 = r(x)
        
        x = tf.keras.layers.add([x1, x2])
        at = BahdanauAttention(128)(x, x, x)
        x = tf.keras.layers.add([x, at])
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        
    x = layers.Concatenate()([x, temp])
    
    at = BahdanauAttention(128)(x, x, x)
    x = tf.keras.layers.add([x, at])
    x = tf.keras.layers.BatchNormalization()(x)
    
    skips = reversed(skips[:-1])
    mid_skips = reversed(mid_skips)


  # Upsampling and establishing the skip connections
    for up, skip, mid_skip in zip(up_stack, skips, mid_skips):
        x = up(x)
        at = BahdanauAttention(128)(x, mid_skip, skip)
        x = tf.keras.layers.add([x, at])
        x = tf.keras.layers.BatchNormalization()(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip, mid_skip])
        at = BahdanauAttention(128)(x, x, x)
        x = tf.keras.layers.add([x, at])
        x = tf.keras.layers.BatchNormalization()(x)

  # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      32, 3, strides=2,
      padding='same')  #64x64 -> 128x128

    x = last(x)
    
    x = layers.LeakyReLU()(x)
    
    at = BahdanauAttention(128)(x, x, x)
    x = tf.keras.layers.add([x, at])
    x = tf.keras.layers.BatchNormalization()(x)
    
    conv_inp1 = ResidualBlock(32,3, 'same')(f_inputs)
    conv_inp2 = ResidualBlock(32,3, 'same')(conv_inp1)
    
    at = BahdanauAttention(128)(x, conv_inp1, conv_inp2)
    x = tf.keras.layers.add([x, at])
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = layers.Concatenate()([x, conv_inp1, conv_inp2])
    
    at = BahdanauAttention(128)(x, x, x)
    x = tf.keras.layers.add([x, at])
    x = tf.keras.layers.BatchNormalization()(x)
    
    initializer = tf.random_normal_initializer(0., 0.02)
    
    x = layers.Conv2D(output_channels, 1, padding='same', kernel_initializer=initializer, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
 
 
 
 
 
model = unet_model(13)
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.load_weights('road_detection5/model.h5')
model.summary()
print ("Model loaded")
# Create a colormap for the labels


# RGB colors of the classes
COLORS = [(80/255, 168/255, 250/255),
          (242/255, 130/255, 30/255),
          (50/255, 50/255, 50/255),
          (27/255, 44/255, 129/255),
          (163/255, 68/255, 222/255),
          (115/255, 0/255, 0/255),
          (255/255, 255/255, 255/255),
          (191/255, 191/255, 191/255),
          (150/255, 150/255, 150/255),
          (22/255, 146/255, 0/255),
          (245/255, 239/255, 46/255),
          (181/255, 103/255, 10/255),
          (235/255, 0/255, 0/255)]

# Number of classes
NUM_CLASSES = 13



def detect_raising_edge (medians_vector):
    
    old_value = medians_vector[0]
    for i in range(len(medians_vector)-1):
        if medians_vector[i] != old_value :
            break ; 
    return(i)        
    
def detect_falling_edge (medians_vector):
    
    ind = detect_raising_edge(medians_vector)
    if (ind != 0):
        for i in range(ind , len(medians_vector)-1):
            if medians_vector[i]== 255 :
                break ; 
        return(i)
    return(0)   


# to reshape the output of the model
width = 960
height = 480



from imutils.video import VideoStream


print("[INFO] starting video stream...")
vs = VideoStream("videos/3.mp4").start()

   
std_array = []
#the counter of initial frames
cnt = 0
average = -1
    
    
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    image = vs.read()
    cnt = cnt + 1
    print(cnt)
    frame_orig = image
    ####################### median vector for storing the medians of line pixels
    medians = []
    
    ####################### reshape for model input
    image = cv2.resize(image,(512,512))
    
    ###################### extract the road
    start_time = time.time()
    prediction_r = model.predict(np.expand_dims(image, axis=0))[0]
    prediction = tf.argmax(prediction_r, axis=2)
    print("time - {}".format(time.time()-start_time))

    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = prediction.numpy().flatten()
    
    image = image.flatten() 
    
    ####################### result to our road segmentation
    res = np.array([image[i] if prediction[i] == 7  else 0 for i in range(len(prediction))])

    res= res.reshape(512,512)
    res = res.astype(np.uint8)
    
    image=cv2.resize(res,(width,height))
        
    ###################### delete the upper half screen, assumption is the road is on the lower half
    crop_img = image[int(height/2):height,:]

    ###################### remove the shadow
    rgb_planes = cv2.split(crop_img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 5)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    image = cv2.merge(result_planes)


    ###################### compute the median of line pixels only if more than half are non zero pixels
    for i in range (int(height/2)):
        # process line by line
        line = np.array(image [i,:])
        #count the non black pixels 
        nb_nonblack_pixels = np.count_nonzero(line)
        # process the line only if more than 50% its pixels are not black
        if nb_nonblack_pixels > int(width/2):
            # remove the black pixels, we need them not to compute the median
            line = line[line != 0]
            medians.append(np.median(line))
        else :
            medians.append(0)
        
    # to remove with video input    
    medians.append(0)
    medians = np.array(medians)
    
    # detect the edges to delete jittering of median vector
    # a scenario for this is , road (median values), no road (0) and then back to road(median values)
    ind_rising_edge = detect_raising_edge(medians)
    ind_falling_edge = detect_falling_edge(medians)
    

    # we need this middle part only
    middle = medians[ind_rising_edge+1:ind_falling_edge-1]
 
    
    # compute the standard deviation of the median vector to detect the jittering
    # in the median vector
    if cnt < 10:
        if (len(middle)!=0):
            std_array.append(np.std(middle))
            print("std ", np.std(middle))
   
   # if there are 10 frames then we compute the average of std dev array

    if cnt == 10:
       average = np.mean(std_array)
       print ("the average is ",average) 
    
    # the average has been computed , we can predict
    if (average != -1):       
        label = "speed_bump" if average < np.std(middle) else "No speed_bump"        
        color = (0, 255, 0) if label == "speed_bump" else (0, 0, 255)
            # display the label and bounding box rectangle on the output
            # frame
        cv2.putText(frame_orig, label, (0, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)   
    
        # save the image for debugging
        outfile = 'output_video_detection/%s_prediction.jpg' % (cnt)
        frame_orig = cv2.resize(frame_orig, (960, 640))   
        cv2.imwrite(outfile,frame_orig)
        
        
        #time.sleep(1)
        print('type here to exit')
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            vs.stop()