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

train_path = r'C:\MachineLearning\DEEP_LEARNING\DL_PROGRAMS\Digits\training'
validation_path = r'C:\MachineLearning\DEEP_LEARNING\DL_PROGRAMS\Digits\validation'

train_batches = ImageDataGenerator().flow_from_directory(train_path , 
                                  target_size = (224,224) , classes=['zero' , 'one' , 'twoother'] ,
                                  batch_size=16)                                                     
validation_batches = ImageDataGenerator().flow_from_directory(validation_path , 
                                       target_size = (224,224) , 
                                       classes=['zero' , 'one' , 'twoother'] , batch_size=8)

image_input = Input(shape=(224, 224, 3))
resnet50_model = keras.applications.resnet50.ResNet50(input_tensor=image_input)

last_layer = resnet50_model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(3, activation='softmax', name='output_layer')(x)
our_model2 = Model(inputs=image_input,outputs= out)

for layer in our_model2.layers[:-1]:
    layer.trainable = False

our_model2.compile(Adam(lr = 0.001) , loss = 'categorical_crossentropy' , 
                   metrics = ['accuracy'])

num_epochs = 40
histogram = our_model2.fit_generator(train_batches , steps_per_epoch=10 ,
                                     validation_data = validation_batches ,
                                     validation_steps = 5 ,epochs = num_epochs ,
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

image = cv2.imread(r'C:\MachineLearning\DEEP_LEARNING\DL_PROGRAMS\Digits\testing\twonettest.png')
#print(image.shape)

image = cv2.resize(image , (224,224))
image = np.expand_dims(image, axis=0)
#print(image.shape)

prediction = our_model2.predict(image)
print(prediction)
max_index = np.argmax(prediction)
if max_index == 0:
    print('zero' , prediction[0][0])
if max_index == 1:
    print('one' , prediction[0][1])
if max_index == 2:
    print('two' , prediction[0][2])

#for prediction in prediction:
#    if prediction[0] > 0.5:
#        print('dog')
#        print('probability of',prediction[0])
#    else :
#        print('cat')
#        print('probability of',prediction[1])
        
plt.show()