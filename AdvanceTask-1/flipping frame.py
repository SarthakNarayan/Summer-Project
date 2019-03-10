import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
 
# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
# Lucas kanade params
lk_params = dict(winSize = (20, 20),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)
 
cv2.namedWindow("Together")
cv2.setMouseCallback("Together", select_point)
draw = False 
point_selected = False
point = ()
old_points = np.array([[]])
mask = np.zeros((480,640,3),np.uint8)
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    if point_selected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
 
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points

        ox,oy = point
        x, y = new_points.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        if draw == True:
            cv2.line(mask,(ox,oy),(x,y),(255,0,0),3)
        point = (x,y)
    
    img = cv2.add(frame,mask)

    cv2.imshow("mask", mask)
    cv2.imshow("Together", img)

    
 
    key = cv2.waitKey(30)

    if key == ord('u') & 0xFF:
        draw = False
        print("pen lifted")
    if key == ord('d') & 0xFF:
        draw = True
        print("pen down")
    if key == ord('c') & 0xFF:
        mask = np.zeros((480,640,3),np.uint8)
        print ("the screen is now clear")
    if key == ord('s') & 0xFF :
        cv2.imwrite('C:\Users\Lenovo\Desktop\open cv programsandimages\savedimageeight.jpg',mask)
        print("the image has been saved")
        cv2.imshow('savedimage',mask)
    if key == 27 & 0xFF:
        break

 
cap.release()
cv2.destroyAllWindows()
