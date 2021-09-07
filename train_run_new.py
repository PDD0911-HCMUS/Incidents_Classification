import os
import numpy as np
from tensorflow import keras
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

    conv_base = EfficientNetB6(weights="imagenet", include_top=False, input_shape=(cfs.IMG_HEIGHT, cfs.IMG_WIDTH, cfs.NUM_COLORS))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    #avoid overfitting
    model.add(layers.Dropout(rate=0.2, name="dropout_out"))
    #model.add(layers.Dropout(dropout_rate=0.2, name="dropout_out"))
    # Set NUMBER_OF_CLASSES to the number of your final predictions.

    model.add(layers.Dense(cfs.NUM_CLASS, activation="softmax", name="fc_out"))
    conv_base.trainable = False

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )

    
    model_checkpoint = ModelCheckpoint('EfficientNetB6.hdf5', monitor='loss',verbose=1, save_best_only=True)

    print("---------Loading Data---------")
    train_images, train_class_name = predata.create_dataset(cfs.DATA_TRAIN_DIR)
    valid_images, valid_class_name = predata.create_dataset(cfs.DATA_TRAIN_DIR)

    train_class_dict={k: v for v, k in enumerate(np.unique(train_class_name))}
    target_val_train=  [train_class_dict[train_class_name[i]] for i in range(len(train_class_name))]
    print(train_class_dict)


    valid_class_dict={k: v for v, k in enumerate(np.unique(valid_class_name))}
    target_val_valid=  [valid_class_dict[valid_class_name[i]] for i in range(len(valid_class_name))]

    print("---------Training---------")
    history = model.fit(
        x = np.array(train_images),
        y = np.array(list(map(int, target_val_train))),
        steps_per_epoch=2000, 
        epochs=15,
        validation_data=(np.array(valid_images), np.array(list(map(int, target_val_valid)))),
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        callbacks=[model_checkpoint]
    )

if __name__=='__main__':
    main()



'''
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=1000,epochs=1,callbacks=[model_checkpoint])
'''
