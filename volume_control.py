# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:14:28 2021

@author: Anurag
"""
'''
https://github.com/AndreMiras/pycaw
'''
import cv2
import time
import numpy as np
import hand_track as ht
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#######################################
w_cam, h_cam = 640, 480
#######################################

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
# current_time = 0
previous_time = 0

##### creating object #########
detector = ht.hand_detect(detect_conf=0.7)
##############################

######## pycaw ################
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate( IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0, None)

min_volume = volume_range[0]
max_volume = volume_range[1]
##############################
vol = 0
vol_bar = 400
vol_percentage = 0

while True:
    ret, img = cap.read()
    if ret:
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img, draw=False)
        if len(landmark_list) !=0:
            # print(landmark_list[4], landmark_list[8])
            x1, y1 = landmark_list[4][1], landmark_list[4][2]
            x2, y2 = landmark_list[8][1], landmark_list[8][2]
            center_x, center_y = (x1 + x2)//2, (y1 +y2)//2
            
            cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (center_x, center_y), 10, (255, 255, 0), cv2.FILLED)
            
            length = math.hypot(x2 - x1, y2 -y1)
            # print(length)
            
            ############ range conversion ###################
            #hand range = 40 <-> 320
            # volume range  -65 <->  0
            vol = np.interp(length, [50, 280], [min_volume, max_volume])
            vol_bar = np.interp(length, [50, 280], [400, 150])
            vol_percentage = np.interp(length, [50, 280], [0, 100])
            # print(int(length), vol)
            
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), cv2.FILLED)
        
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(vol_percentage)), (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        
        
        ###### FPS ########
        current_time = time.time()
        fps = 1/(current_time- previous_time)
        previous_time = current_time
        ###################
        
        cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        
        cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
        cv2.imshow('result', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
    else:
        break
    
    
cv2.destroyAllWindows()
        
    