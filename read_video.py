# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 03:50:14 2019

@author: HP
"""

import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if(ret==False):
        continue
    cv2.imshow("Hey! Your Video :p",frame)
    k=cv2.waitKey(1) & 0xFF
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    