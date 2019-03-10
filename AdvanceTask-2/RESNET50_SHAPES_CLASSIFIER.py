import numpy as np
import keras
import cv2
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers import Input
from keras.models import Model
from matplotlib import pyplot as plt

train_path = r'E:\RMI project\AdvanceTask-2\SHAPES\training'
validation_path = r'E:\RMI project\AdvanceTask-2\SHAPES\validation'

train_batches = ImageDataGenerator().flow_from_directory(train_path , 
                                  target_size = (224,224) , classes=['circle' ,'rectangle' ,'triangle'] ,
                                  batch_size=6)                                                     
validation_batches = ImageDataGenerator().flow_from_directory(validation_path , 
                                       target_size = (224,224) , 
                                       classes=['circle' ,'rectangle' ,'triangle'] , batch_size=4)

image_input = Input(shape=(224, 224, 3))
resnet50_model = keras.applications.resnet50.ResNet50(input_tensor=image_input)

last_layer = resnet50_model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(3, activation='softmax', name='output_layer')(x)
our_model2 = Model(inputs=image_input,outputs= out)

# leaving the layers untrainable as I got a very high accuracy
#for layer in our_model2.layers[:-1]:
#    layer.trainable = False

our_model2.compile(Adam(lr = 0.001) , loss = 'categorical_crossentropy' , 
                   metrics = ['accuracy'])

num_epochs = 40
histogram = our_model2.fit_generator(train_batches , steps_per_epoch=5 ,
                                     validation_data = validation_batches ,
                                     validation_steps = 3 ,epochs = num_epochs ,
                                     verbose = 1)

train_loss=histogram.history['loss']
val_loss=histogram.history['val_loss']
train_acc=histogram.history['acc']
val_acc=histogram.history['val_acc']
xc=range(num_epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

print('saving the model')
our_model2.save(r'E:\RMI project\AdvanceTask-2\shapesclassifiertrained.h5')
print('model saved')

plt.show()
