import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np

#For non bright images
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=3,
        fill_mode='nearest')

img = load_img(r'C:\MachineLearning\DEEP_LEARNING\DL_PROGRAMS\SHAPES\testing\savedimage4.jpg')  
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=r'C:\MachineLearning\DEEP_LEARNING\DL_PROGRAMS\SHAPES\training\triangle',     
                          save_prefix='one', save_format='jpeg'):
    i += 1
    if i > 15:
        break  # otherwise the generator would loop indefinitely