#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:34:37 2017

@author: tapos-datta
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 23:29:24 2017

@author: Tapos
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('th')
import os , datetime

# Import datasets, classifiers and performance metrics


seed=7
np.random.seed(seed)

"""
    #define the larger model
    
    Convolutional layer with n feature maps of size 5×5.
    Pooling layer taking the max over 2*2 patches.
    Convolutional layer with m feature maps of size 3×3.
    Pooling layer taking the max over 2*2 patches.
    Dropout layer with a probability of 35%.
    Flatten layer.
    Fully connected layer with N1 neurons and rectifier activation.
    Fully connected layer with N2 neurons and rectifier activation.
    Output layer.
    
"""

def larger_model():
    
    # create model
    
    model = Sequential()
    model.add(Conv2D(70, (5, 5), input_shape=(1, 28, 28), activation="relu", padding="valid",strides=1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Conv2D(125, (3,3), activation="relu",strides=1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu',use_bias=True,bias_initializer='zeros'))
    model.add(Dropout(0.35))
    model.add(Dense(150, activation='relu', use_bias=True,
                      bias_initializer='zeros'))
    model.add(Dropout(0.35))
    model.add(Dense(num_classes, activation='softmax'))  
    
    # Compile model
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# set directory path to load dataset

orig_path = os.getcwd() +'/Converted_numpy_arrays/'

X = []
t = []
indx = []
cnt = 0

#load normalized images through numpy arrays of image
  
for i in range(1,85):
    if i==63 or i==65 or i==77 or i==72:
        continue
    tmp = np.loadtxt(orig_path+str(i)+'.txt')
    for j in range(tmp.shape[0]):
        X.append(tmp[j])
        t.append(i-1)
        indx.append(cnt)
        cnt += 1
#    print("Line 32, has been read to: ", i)

# store all example to X and corresponding target to t

X = np.asarray(X)
t = np.asarray(t)

# Random shuffle the arrays.
X = list(zip(X,t,indx))
np.random.shuffle(X)
X,t,indx = zip(*X)
X = np.asarray(X)
t = np.asarray(t).ravel()
indx = np.asarray(indx)

#divided dataset into training and validation part by 90:10

train_x = X[0:146664]
train_y = t[0:146664]
val_x = X[146664:]
val_y = t[146664:]
train_indx = indx[0:146664]
val_indx = indx[146664:]


# reshape to be [samples][pixels][width][height]

train_x = train_x.reshape(train_x.shape[0], 1, 28, 28).astype('float32')
val_x = val_x.reshape(val_x.shape[0], 1, 28, 28).astype('float32')

#normalize inputs from 0-255 to 0-1
train_x = train_x / 255
val_x = val_x / 255

# one Hot encode output
train_y = np_utils.to_categorical(train_y)
normal_y = val_y
val_y = np_utils.to_categorical(val_y)

# number of classes after one Hot encode
num_classes = val_y.shape[1]

#print("train_x: ", train_x.shape, " train_y: ", train_y.shape
#		, " valx_: ", val_x.shape, " val_y: ", val_y.shape)
#print("unique in train: ", len(np.unique(train_y)), " unique in vali: ", len(np.unique(val_y)))
#print("Okay!!!!!")


# build the model
model = larger_model()
    
# Fit the model
model.fit(train_x, train_y, validation_data=(val_x,val_y),epochs=100, batch_size=200, verbose=1)
# predict examples
predict = model.predict(val_x, batch_size=1940)


print("predict.shape: ", predict.shape)
#Show the false predicted examples with actual targets
for i in range(len(predict)):
    if(np.argmax(predict[i]) != np.argmax(val_y[i])):
        print("File number: ", val_indx[i], " output: ", 
              np.argmax(predict[i]), " label: ", np.argmax(val_y[i]))

