#FINAL PROJECT: HOLISTIC FACE AND BODY DETECTION

import cv2
import mediapipe as mp

# mediapipe:  It can be used to make cutting-edge Machine Learning Models like face detection, multi-hand tracking, object detection, and tracking, and many more

#these are drawing utilites which is used draw dots and lines 
mp_drawing=mp.solutions.drawing_utils

#mp_drawing=mp.solutions.drawing_utils
#under this we can able to access Face Detection Face Mesh, Iris, Hands ,Pose ,Holistic ,Selfie Segmentation,Hair Segmentation et

mp_drawing_styles=mp.solutions.drawing_styles #this one is style
mp_holistic=mp.solutions.holistic #it has the abiity to plot like stickmen

cap= cv2.VideoCapture('c:\\Users\\user\\OneDrive\\Desktop\\OPENCV images\\UFC.mp4')
#cap= cv2.VideoCapture(0)
background=cv2.VideoCapture('c:\\Users\\user\\OneDrive\\Desktop\\OPENCV images\\Star background.mp4')


# To specify the holistic levels like sensitivity of feature detection
# if sensitivity is high then may get false feature
# if sensitivity is low then may some features be missed 
# soo 0.5 is good to keep

# 'with' is used for resource management 
# refer diss the txt document
with mp_holistic.Holistic(
     min_detection_confidence=0.5,
     min_tracking_confidence= 0.5
     )as holistic:

    while cap.isOpened():
        _,image=cap.read()
        _,backgroundimage=background.read()
        
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result= holistic.process(image)
        image= cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


#Face recognition: calling mp_drawing=mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            backgroundimage,
            result.face_landmarks,
            #landmark_drawing_spec_=None #it shows only 4 or 5 co-ordinates instead of 52 co-ordinates
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())  
        
#Body recognition:
        mp_drawing.draw_landmarks(
            backgroundimage,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow(" stick show", cv2.flip(backgroundimage,1)) 
        key=cv2.waitKey(1)
        if key==ord('q'):
            break

cap.release()


