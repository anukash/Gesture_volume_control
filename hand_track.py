"""
Created on Wed Sep 22 13:33:51 2021

@author: Anurag
"""
'''
https://google.github.io/mediapipe/solutions/hands
Hand tracking uses two main module at the backend
palm detection and hand landmarks detection
palm detection work on complete image and it provide cropped image of hand
hand landmark finds 25 landmarks in cropped image
'''

import cv2
import mediapipe as mp
import time

class hand_detect():
    def __init__(self, mode=False, max_hand=2, detect_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hand = max_hand
        self.detect_conf = detect_conf
        self.track_conf = track_conf
        
        # static_image_mode=False,
        # max_num_hands=2,
        # min_detection_confidence=0.5,
        # min_tracking_confidence=0.5 
    
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hand, self.detect_conf, self.track_conf)
        self.hand_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        # print(results.multi_hand_landmarks)
        
        #check multiple hands
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:        
                    self.hand_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
        return img
    
    
    def find_position(self, img, hand_number=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            
            for i_d, l_m in enumerate(my_hand.landmark):   #location will come in decimal so convert them into pixel
              h, w, c = img.shape
              center_x, center_y = int(l_m.x*w), int(l_m.y*h)
              # print(i_d, center_x, center_y)
              landmark_list.append([i_d, center_x, center_y])
              if draw:
                  cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), cv2.FILLED)
        return landmark_list



def main():
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector = hand_detect()
    while True:
        ret, img = cap.read()
        
        if ret:
            img = detector.find_hands(img)
            landmark_list = detector.find_position(img, hand_number=0, draw=True)
            if len(landmark_list) != 0:
                print(landmark_list[4])
            
            current_time = time.time()
            fps = 1/(current_time- previous_time)
            previous_time = current_time
            
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3 )
            cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
            cv2.imshow('result', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break


if __name__=="__main__":
    main()
    cv2.destroyAllWindows()