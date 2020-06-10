# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 04:17:15 2019

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 03:50:14 2019

@author: HP
"""

import cv2
import numpy as np
cap=cv2.VideoCapture(0)
skip=0
fdata=[]
fsec=[]
dataset_path='C:/Users/HP/Desktop/ml/Project_5_Real_Time_Face_Recognition/data'
file_name=input("Enter name of the person: ")
#fc=cv2.CascadeClassifier(r'C:\Users\HP\Desktop\ml\Project_5_Real_Time_Face_Recognition\haarcascade_frontalface_alt.xml')
fc=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    #gf=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if(ret==False):
        continue
    face=fc.detectMultiScale(frame,1.3,5)
    
    for f in face:
        x,y,w,h=f
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        offset=10
        fsec=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        fsec=cv2.resize(fsec,(100,100))
    #cv2.imshow("Hey! Your Video :p",gf)
    
    cv2.imshow("Hey! Your Video :p",frame)
    cv2.imshow("Face Section :p",fsec)
    
    skip+=1
    if(skip%10==0):
        fdata.append(fsec)
        print(len(fdata))  
    k=cv2.waitKey(1) & 0xFF
    if k==ord('q'):
        break     
fdata=np.asarray(fdata)
fdata=fdata.reshape((fdata.shape[0],-1))
print(fdata.shape)
np.save(dataset_path+file_name+'.npy',fdata)  
print("Data Saved")
cap.release()
cv2.destroyAllWindows()
    