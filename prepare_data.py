import os
import config_constants as cfs
from PIL import Image, ImageOps
import numpy as np
from time import sleep
import cv2



def Get_Images(data_dir, sort = True):
    """For a given path, returns a (sorted) list containing all
    files."""
    image_paths = []
    for folder, _, imgs in os.walk(data_dir):
        for image_path in imgs:
            image_paths.append(os.path.join(folder, image_path))
    if sort is True:
        image_paths = sorted(image_paths)
    return image_paths

def Read_All_Images(images_path, num_colors):
    list_images = []
    for i, file in enumerate(images_path):
        current_img = Image.open(file)
        if num_colors == 3 and current_img.mode in ['L', 'P']: #Stopgap for B&W images
            bw_image = current_img
            current_img = Image.new("RGB", current_img.size)
            current_img.paste(bw_image)              
        current_img = np.asarray(current_img)
        list_images.append(current_img)
        
    data_images = np.array(list_images)
    print(len(data_images))
    return data_images

def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (cfs.IMG_HEIGHT, cfs.IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


