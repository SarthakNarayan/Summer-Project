import cv2
import numpy as np
from keras.models import load_model

print("loading the model")
our_model2 = load_model(r'E:\RMI project\AdvanceTask-2\shapesclassifiertrained.h5')
print('model loaded')

path = r'E:\RMI project\AdvanceTask-2\SHAPES\testing\savedimage'

for i in range(1,5):
    image_path = path + str(i) + '.jpg'
    image = cv2.imread(image_path)
    image = cv2.resize(image , (224,224))
    image = np.expand_dims(image, axis=0)

    prediction = our_model2.predict(image)
    print(prediction)
    max_index = np.argmax(prediction)
    if max_index == 0:
        print('circle' , prediction[0][0])
    if max_index == 1:
        print('rectangle' , prediction[0][1])
    if max_index == 2:
        print('triangle' , prediction[0][2])
