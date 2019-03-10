import cv2
import numpy as np

# to start the camera and use webcam
cap = cv2.VideoCapture(0)

#naming a window for object tracking
cv2.namedWindow('SettingHSV')

#defining the nothing function
def nothing():
    pass

blackbackground = np.zeros((512,512,3),np.uint8)
while(True):
    i=-1
    #capture frame by frame 
    ret,frames = cap.read()

    #blurring the image
    kernelforblurring = np.ones((15,15),np.float32)/225
    blur = cv2.filter2D(frames,-1,kernelforblurring)

    #converting brg to hsv
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    #define range of the color object
    lower_blue = np.array([159,154,86])
    upper_blue = np.array([255,255,241])

    #thresholding the image to get only white colors by creating a mask
    mask = cv2.inRange(hsv,lower_blue,upper_blue)

    #performing morphological operations on mask
    kernelforerosion = np.ones((5,5) , np.uint8)
    erodedmask = cv2.erode(mask,kernelforerosion,iterations = 1)

    #using bitwiseand to do the mask
    result = cv2.bitwise_and(blur,blur,mask = erodedmask)

    #for finding and drawing contours
    _,contours,_ = cv2.findContours(erodedmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print len(contours)
    for contour in contours :
        i=i+1
        area = cv2.contourArea(contour)
        if area > 3500:
            frameswithcontours = cv2.drawContours(result,contours,i,(0,255,0),4)
           
    #getting the centoid of the image
    if(len(contours) != 0) :
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            draw = 1
        else:
            draw = 0
        
        if (draw == 1):
            cv2.circle(result,(cX,cY),5,(255,0,0),-1)
            cv2.line(blackbackground,(cX,cY),(cX+1,cY+1),(255,0,0),10)
    else :
        print("no contour found")

    #displaying the video
    cv2.imshow('SettingHSV',mask)
    cv2.imshow('resultant',result)
    cv2.imshow('linedrawing',blackbackground)

    #setting esc key to end the process
    k=cv2.waitKey(1) & 0xFF
    if k == 27 :
        break

#to release camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

    
    
    
