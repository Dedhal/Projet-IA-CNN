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
validation_file = open(USE_DATASET + "test\\labels.json")

train_data = json.load(train_file)
validation_data = json.load(validation_file)

if USE_DATASET != "G:\\Datasets\\Cifar100\\":
    print(str(len(train_data["annotations"])))
    
    df = pd.DataFrame.from_dict(train_data["annotations"])
    
    dataset_len = len(df["image_id"].unique())

else:
    df = pd.DataFrame.from_dict([train_data["labels"]])
    dataset_len = 50000

df = df.T

df.reset_index(inplace=True)
df = df.rename(columns={"index":"image_id", 0:"category_id"})

print(df)
print(df.head())
