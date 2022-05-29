import cv2
import numpy as np
import os
os.chdir(R"C:\Users\Home-PC\PycharmProjects\CarDetection")

def pega_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture("test2.mp4") #importing the test video

BGS = cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=50, detectShadows=True) #video segmentation

while True:
    ret, frame1 = cap.read()    #keeps the video going untill no frames avail

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(grey, (9, 9), 5)     #reducing the noice in the video
    img_sub = BGS.apply(blur)    #makes the background black and takes the moving objects and turn them to white objects for easy detection
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))     #helps in removing the black noice in cars in the video after coloring them in white

    cnt, h = cv2.findContours(dilat, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)



    for(i, c) in enumerate(cnt):
        (x, y, w, h) = cv2.boundingRect(c)

        vld_cnt = (100<w<180) and (100<h<180) # limitation to the objects detected based on its size
        if not vld_cnt:
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2) #drawing a rectangle to show the detected cars
        center = pega_center(x, y, w, h) #marking the center of the marking rectangle

        cv2.circle(frame1, center, 4, (0, 0, 255), 1)
    cv2.imshow("video original", frame1)
        #cv2.imshow("dilat", dilat)
        #cv2.imshow("img_sub", img_sub)

    if cv2.waitKey(1) == 'q':
        break

cv2.destroyAllWindows()
cap.release()