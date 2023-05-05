import json

import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import tensorflow as tf
import cv2
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import warnings

train_file = open("G:\\Datasets\\CoCo\\train\\labels.json")
validation_file = open("G:\\Datasets\\CoCo\\validation\\labels.json")

train_data = json.load(train_file)
validation_data = json.load(validation_file)

def process_image(img):
    downsample_size = 200
    img_read = io.imread(img)

    try:
        img_read = cv2.cvtColor(img_read,cv2.COLOR_GRAY2RGB)

    except:
        pass

    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    

    img_read = img_read
    img_read = tf.expand_dims(img_read, axis=0)

    
    return img_read

imgs = []
labels = []

print(str(len(train_data["annotations"])))

df = pd.DataFrame.from_dict(train_data["annotations"])

print(df["image_id"].unique()[961])
print(df["image_id"].unique()[962])
print(df["image_id"].unique()[963])


df_validation = pd.DataFrame.from_dict(validation_data["annotations"])

def one_hot_converter(liste):
    res = []
    for i in range(91):
        if i in liste:
            res.append(1.0)
        else:
            res.append(0.0)

    return res

def train_gen(datas, batchsize):
    for i, img in enumerate(datas["image_id"].unique()):
        x = process_image("G:\\Datasets\\CoCo\\train\\data\\" + str("%012d" % (img,)) + ".jpg")
        y = one_hot_converter(list(set(datas[datas["image_id"] == img]["category_id"].to_list())))
        y = np.array(y).reshape(1, 91)
        yield x, y

def validation_gen(datas, batchsize):
    for i, img in enumerate(datas["image_id"].unique()):
        x = process_image("G:\\Datasets\\CoCo\\validation\\data\\" + str("%012d" % (img,)) + ".jpg")

        yield  x

# Model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), input_shape=(200,200,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(256, (3,3)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))
model.add(Dense(91, activation='softmax'))



# Compile the Model

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

history = model.fit(train_gen(df, 10), batch_size=10, epochs=25, validation_data = validation_gen(df_validation,10), verbose = 1, validation_steps=3)