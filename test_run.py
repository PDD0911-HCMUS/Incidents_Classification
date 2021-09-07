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
from tensorflow.keras import models

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
model = models.Sequential()
model = load_model('EfficientNetB6_20210709_5h58am.hdf5')
model.summary()

for item in os.listdir(path_test):
    #print(os.path.join(path_test, item))
    img = io.imread(os.path.join(path_test, item))
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img,(1,)+img.shape)
    results = model.predict_generator(img,1,verbose=0)
    res = np.reshape(results, results.shape[1])

    max_value = max(res)
    max_index = res.tolist().index(max_value)
    print(item, " -- " , classes[max_index])



# def testGenerator(test_path,num_image = 1,target_size = (224,224),flag_multi_class = False,as_gray = True):
#     for i in range(num_image):
#         img = io.imread(os.path.join(test_path,"%d.jpg"%i))
#         img = img / 255
#         img = trans.resize(img,target_size)
#         img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#         img = np.reshape(img,(1,)+img.shape)
#         return img

# testGene = testGenerator("incidents_cleaned/test_temp")
# print(testGene.shape)


# results = model.predict_generator(testGene,1,verbose=1)
# res = np.reshape(results, results.shape[1])

# print('class: ', res)
# max_value = max(res)
# max_index = res.tolist().index(max_value)
# print('value predict: ', max_value, '- Type: ', max_index, '- Type name: ', classes[max_index])