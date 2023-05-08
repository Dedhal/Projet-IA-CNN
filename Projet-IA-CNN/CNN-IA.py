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
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
import warnings
import generator as gen

from PIL import Image
import pickle

USE_DATASET = gen.USE_DATASET

train_file = open(USE_DATASET + "train\\labels.json")
if USE_DATASET != gen.CIFAR100:
    validation_file = open(USE_DATASET + "validation\\labels.json")
else:
    validation_file = open(USE_DATASET + "test\\labels.json")

train_data = json.load(train_file)
validation_data = json.load(validation_file)

if USE_DATASET != gen.CIFAR100:
    print(str(len(train_data["annotations"])))
    
    df = pd.DataFrame.from_dict(train_data["annotations"])
    
    dataset_len = len(df["image_id"].unique())

else:
    df = pd.DataFrame.from_dict([train_data["labels"]])
    dataset_len = 50000

    df = df.T
    
    df.reset_index(inplace=True)
    df = df.rename(columns={"index":"image_id", 0:"category_id"})


res = [0,0,0]
try:
    with open(gen.MEAN_FILE, 'rb') as f:
        means = pickle.load(f)
    
except:
    for i, image in enumerate(df["image_id"].unique()):
        if USE_DATASET != gen.CIFAR100:
            img_read = io.imread(USE_DATASET + "train\\data\\" + str("%012d" % (image,)) + ".jpg")
        else:
            img_read = io.imread(USE_DATASET + "train\\data\\" + str(image) + ".jpg")

        try:
            img_read = cv2.cvtColor(img_read,cv2.COLOR_GRAY2RGB)

        except:
            pass

        img_read = transform.resize(img_read, (gen.DIM, gen.DIM), mode='constant')
        img_read = np.array(img_read)
        img_read = np.sum(img_read, axis=1)
        img_read = np.sum(img_read, axis=0)
        res = np.add(res, img_read)

        if i%100 == 0:
            print(str(i) + " / " + str(dataset_len) + " images completed")

    div = gen.DIM * gen.DIM * dataset_len
    means = np.divide(res, div)

    with open(gen.MEAN_FILE, 'wb') as f:
        pickle.dump(means,f)

#input("Press Enter to continue...")
if USE_DATASET != gen.CIFAR100:
    df_validation = pd.DataFrame.from_dict(validation_data["annotations"])

else:
    df_validation = pd.DataFrame.from_dict([validation_data["labels"]])
    df_validation = df_validation.T
    
    df_validation.reset_index(inplace=True)
    df_validation = df_validation.rename(columns={"index":"image_id", 0:"category_id"})

print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

#input()

#l2_reg = 0.001
# Model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), input_shape=(224,224,3), activation='relu', padding='same'))
#model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
#model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
#model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
#model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
#model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4096,activation='relu'))
model.add(Dense(1000))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(5, activation='softmax'))

# Compile the Model

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

train_generator = gen.DataGenerator(df)
validation_generator = gen.DataGenerator(df_validation, mode="validation")

#history = model.fit(train_gen(df, 1), batch_size=32, steps_per_epoch = 500, epochs=25, validation_data = validation_gen(df_validation,1), verbose = 1, validation_steps=3)
history = model.fit(train_generator, batch_size=gen.BATCH_SIZE, steps_per_epoch = 8, epochs=40, validation_data = validation_generator, verbose = 1, validation_steps=3)

# summarize history for accuracy

models = os.scandir("G:\\Datasets\\CoCo\\models\\")
i = len(list(models))
model.save("G:\\Datasets\\CoCo\\models\\baby_cnn_"+str(i)+".h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("G:\\Datasets\\CoCo\\models\\evals\\baby_cnn_"+str(i)+"_acc.png")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("G:\\Datasets\\CoCo\\models\\evals\\baby_cnn_"+str(i)+"_loss.png")