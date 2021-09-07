import re
from keras.backend import reshape
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import config_constants as cfs
import cv2

path_test = os.path.join(cfs.DATA_TEST_DIR, 'crash')
#path_test = cfs.DATA_TEST__TEMP_DIR
target_size = (224,224)

classes = [ 'animals',
            'collapse',
            'crash',
            'fire',
            'flooding',
            'landslide',
            'snow',
            'treefall']

model = load_model('EfficientNetB6_20210809_2h34am.hdf5')
model.summary()

for image in os.listdir(path_test):
    image_path= os.path.join(path_test, image)
    image_test = cv2.imread( image_path, cv2.COLOR_BGR2RGB)
    image_test=cv2.resize(image_test, (cfs.IMG_HEIGHT, cfs.IMG_WIDTH),interpolation = cv2.INTER_AREA)
    image_test=np.array(image_test)
    image_test = image_test.astype('float32')
    image_test /= 255 
    image_test = np.reshape(image_test,(1,)+image_test.shape)
    results = model.predict(image_test,verbose=0)
    res = np.reshape(results, results.shape[1])

    max_value = max(res)
    max_index = res.tolist().index(max_value)
    print(classes[max_index])



# for item in os.listdir(path_test):
#     #print(os.path.join(path_test, item))
#     img = io.imread(os.path.join(path_test, item))
#     img = img / 255
#     img = trans.resize(img, target_size)
#     img = np.reshape(img,(1,)+img.shape)
#     results = model.predict_generator(img,1,verbose=0)
#     res = np.reshape(results, results.shape[1])

#     max_value = max(res)
#     max_index = res.tolist().index(max_value)
#     print(classes[max_index])