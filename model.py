from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tensorflow.keras import optimizers
import tensorflow as tf
import config_constants as cfs

#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, ... up to  7
# Higher the number, the more complex the model is. and the larger resolutions it  can handle, but  the more GPU memory it will need
# loading pretrained conv base model
#input_shape is (height, width, number of channels) for images

# conv_base = EfficientNetB6(weights="imagenet", include_top=False, input_shape=input_shape)
conv_base = EfficientNetB6(weights="imagenet", include_top=False, input_shape=(cfs.IMG_HEIGHT, cfs.IMG_WIDTH, cfs.NUM_COLORS))

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
#avoid overfitting
model.add(layers.Dropout(dropout_rate=0.2, name="dropout_out"))
# Set NUMBER_OF_CLASSES to the number of your final predictions.
model.add(layers.Dense(cfs.NUM_CLASS, activation="softmax", name="fc_out"))
conv_base.trainable = False