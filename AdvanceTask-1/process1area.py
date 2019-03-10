import cv2
import numpy as np
from matplotlib import pyplot as plt

savedImage = cv2.imread('E:\RMI project\AdvanceTask-1\images\savedimage1.jpg',0)
savedImage2 = cv2.imread('E:\RMI project\AdvanceTask-1\images\savedimage1.jpg',1)

#creating 2 black screen for drawing circle and rectangle
img1 = np.zeros((480,640,3),np.uint8)
img2 = np.zeros((480,640,3),np.uint8)


#drawing circle and rectangle and converting them to grayscale for thresholding
circle = cv2.circle(img2 , (320,320) , 56 ,(255,0,0),5)
circle1 = cv2.cvtColor(circle , cv2.COLOR_BGR2GRAY)

rectangle = cv2.rectangle(img1 , (150,150) , (250,250) , (255,0,0),3)
rectangle1 = cv2.cvtColor(rectangle , cv2.COLOR_BGR2GRAY)

#thresholding and finding contours for image,circle and rectangle

ret , threshold1 = cv2.threshold(savedImage , 125 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_,contours1,_ = cv2.findContours(threshold1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print("the number of contours in figure are ",len(contours1))
cnt1 = contours1[0]
area1 = cv2.contourArea(cnt1)

ret , threshold2 = cv2.threshold(rectangle1 , 125 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_,contours2,_ = cv2.findContours(threshold2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print("the number of contours in square are ",len(contours2))
cnt2 = contours2[0]
area2 = cv2.contourArea(cnt2)

ret , threshold3 = cv2.threshold(circle1 , 125 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_,contours3,_ = cv2.findContours(threshold3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print("the number of contours in circle are ",len(contours3))
cnt3 = contours3[0]
area3 = cv2.contourArea(cnt3)

#finding the minimum enclosing circle area for image, rectangle and circle
(x1,y1) , radius1 = cv2.minEnclosingCircle(cnt1)
center1 = (int(x1) , int(y1))
radius1 = int(radius1)
#savedImage2 = cv2.circle(savedImage2,center1,radius1,(0,0,255),2)
enclosing_circle_area1 = 3.14*radius1*radius1

(x2,y2) , radius2 = cv2.minEnclosingCircle(cnt2)
center2 = (int(x2) , int(y2))
radius2 = int(radius2)
#rectangle = cv2.circle(rectangle,center2,radius2,(0,0,255),2)
enclosing_circle_area2 = 3.14*radius2*radius2

(x3,y3) , radius3 = cv2.minEnclosingCircle(cnt3)
center3 = (int(x3) , int(y3))
radius3 = int(radius3)
#circle = cv2.circle(circle,center3,radius3,(0,0,255),2)
enclosing_circle_area3 = 3.14*radius3*radius3




difference1 = abs(area1-enclosing_circle_area1)
difference2 = abs(area2-enclosing_circle_area2)
difference3 = abs(area3-enclosing_circle_area3)
print("difference for image",difference1)
print("difference for square",difference2)
print("difference for circle",difference3)


if (abs(difference1-difference2)) > (abs(difference3-difference1)):
    print("it is a circle")
else :
    print("it is a square")



cv2.waitKey(0)
cv2.destroyAllWindows()
