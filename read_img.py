# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 02:38:02 2019

@author: HP
"""
import cv2
img=cv2.imread(r'C:\Users\HP\Desktop\ml\Project_5_Real_Time_Face_Recognition\ashi.jpeg')
cv2.imshow('Ashi',img)
cv2.waitKey(0) #time in ms
cv2.destroyAllWindows()