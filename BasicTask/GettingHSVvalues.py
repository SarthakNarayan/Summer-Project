import cv2
import numpy as np

# to start the camera and use webcam
cap = cv2.VideoCapture(0)

#naming a window for object tracking
cv2.namedWindow('SettingHSV')

#defining the nothing function
def nothing():
    pass

#creating trackbars
cv2.createTrackbar('h1w','SettingHSV',0,255,nothing)
cv2.createTrackbar('s1w','SettingHSV',0,255,nothing)
cv2.createTrackbar('v1w','SettingHSV',0,255,nothing)
cv2.createTrackbar('h2w','SettingHSV',0,255,nothing)
cv2.createTrackbar('s2w','SettingHSV',0,255,nothing)
cv2.createTrackbar('v2w','SettingHSV',0,255,nothing)

while(True):

    #capture frame by frame 
    ret,frames = cap.read()

    #blurring the image
    kernelforblurring = np.ones((15,15),np.float32)/225
    blur = cv2.filter2D(frames,-1,kernelforblurring)

    #converting brg to hsv
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    #getting the input from the trackbars
    h1w=cv2.getTrackbarPos('h1w','SettingHSV')
    s1w=cv2.getTrackbarPos('s1w','SettingHSV')
    v1w=cv2.getTrackbarPos('v1w','SettingHSV')
    h2w=cv2.getTrackbarPos('h2w','SettingHSV')
    s2w=cv2.getTrackbarPos('s2w','SettingHSV')
    v2w=cv2.getTrackbarPos('v2w','SettingHSV')

    #define range of the color object
    lower_blue = np.array([h1w,s1w,v1w])
    upper_blue = np.array([h2w,s2w,v2w])

    #thresholding the image to get only white colors by creating a mask
    mask = cv2.inRange(hsv,lower_blue,upper_blue)

    #performing morphological operations on mask
    kernelforerosion = np.ones((5,5) , np.uint8)
    erodedmask = cv2.erode(mask,kernelforerosion,iterations = 1)

    #using bitwiseand to do the mask
    result = cv2.bitwise_and(blur,blur,mask = erodedmask)

    #for finding and drawing contours
    _,contours,_ = cv2.findContours(erodedmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    frameswithcontours = cv2.drawContours(result,contours,-1,(0,255,0),4)

    #getting the centoid of the image
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    #drawing a circle at the centroid of the shape
    cv2.circle(result,(cx,cy),5,(0,0,255),-1)

    #displaying the video
    cv2.imshow('SettingHSV',mask)
    cv2.imshow('resultant',result)

    #setting esc key to end the process
    k=cv2.waitKey(1) & 0xFF
    if k == 27 :
        break

#to release camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

    
    
    
