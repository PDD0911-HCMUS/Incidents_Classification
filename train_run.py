import os
import numpy as np
import config_constants as cfs
import prepare_data as predata
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tensorflow.keras import optimizers
import tensorflow as tf
import config_constants as cfs
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def main():

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
    model.add(layers.Dropout(rate=0.2, name="dropout_out"))
    #model.add(layers.Dropout(dropout_rate=0.2, name="dropout_out"))
    # Set NUMBER_OF_CLASSES to the number of your final predictions.

    model.add(layers.Dense(cfs.NUM_CLASS, activation="softmax", name="fc_out"))
    conv_base.trainable = True

    # I love the  ImageDataGenerator class, it allows us to specifiy whatever augmentations we want so easily...
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Note that the validation data should not be augmented!
    #and a very important step is to normalise the images through  rescaling
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        cfs.DATA_TRAIN_DIR,
        # All images will be resized to target height and width.
        target_size=(cfs.IMG_HEIGHT, cfs.IMG_WIDTH),
        batch_size=cfs.BATCH_SIZE,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode="categorical",
    )

    validation_generator = test_datagen.flow_from_directory(
        cfs.DATA_VALIDATE_DIR,
        target_size=(cfs.IMG_HEIGHT, cfs.IMG_WIDTH),
        batch_size=cfs.BATCH_SIZE,
        class_mode="categorical",
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.RMSprop(lr=2e-5),
        metrics=["acc"],
    )

    
    model_checkpoint = ModelCheckpoint('EfficientNetB6.hdf5', monitor='loss',verbose=1, save_best_only=True)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000, 
        epochs=15,
        validation_data=validation_generator,
        validation_steps=50,
        verbose=1,
        use_multiprocessing=True,
        workers=4,
        callbacks=[model_checkpoint]
    )

if __name__=='__main__':
    main()



'''
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=1000,epochs=1,callbacks=[model_checkpoint])
'''
