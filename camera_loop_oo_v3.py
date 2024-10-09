# -*- encoding: utf-8 -*-

import os
from multiprocessing import Queue #, Manager

import numpy as np
import cv2 as cv

from img_transfer import ImgTransfer
from face_detection_yunet_oo_v3 import FaceDetection 
from face_tracking_oo import FaceTracking
from trajectory import Trajectory 
from mode import Mode
from uart import UART    # Uses pyserial for serial communication
import file_management as fl    

# ===================================================================             
# Camera loop with face detection, face tracking. 
# It sends coord to servo via serial port, 
# and put face images in a queue for face recognition. 
# ===================================================================             
def cameraLoop(imgTransfer: ImgTransfer, resultQueue: Queue):   
    
    if hasPrivileges():
        os.nice(-10)  # High priority task required elevated privileges (sudo)
    
    ifSendData = False
    ifSaveVideo = False 
     
    video = cv.VideoCapture(0)
    faceDetection = FaceDetection(video) 
    faceTracking = FaceTracking()
    
    # Declare trajectories objects for both modes, including the Kalman filters
    traject = {mode: Trajectory(mode) for mode in ['detection', 'tracking']}
    
    imgTransfer.createSharedMemory(faceDetection.frameSize) # shared_memory for image transfer 
    
    if ifSendData: uart = UART()               # Serial communication with microcontroller
    
    # mode: 'faceDetection' OR 'faceTracking 
    # When a face is selected during faceDetection, then faceTracking 
    #                                               is activated after a short laps of time.
    mode = Mode('faceDetection')
    
    while video.isOpened():
        isActive, img = video.read()
        if not isActive:
            print('Camera not active!')
            break
        # Tap any key to exit the loop    
        if cv.waitKey(1) > 0:
            print('Exit the camera loop')
            break
        
        if mode.isInDetectionMode(): 
            faces, largestFaceIndex = faceDetection.detect(img)    #faces: List of Face objects
            if faceDetection.isSuccessful and faces is not None:                        
                activeFace = faces[largestFaceIndex]  # a face object    
                traject['detection'].appendObs(activeFace.observedCenter)
                if traject['detection'].isAtFirstStep(): 
                    traject['detection'].filter.setKalmanInitialState(*activeFace.observedCenter)  # 
                traject['detection'].updateFilter()
                activeFace.smoothCenter = traject['detection'].getLastSmoothPt() 
                
                # Tell the face recognition task if it has to run
                hasToRunRecognition = faceDetection.recognitionCondition(traject['detection'])
        
            if  mode.isTimeToSwitchToTracking(faces):
                faceTracking.initTracker(img, activeFace.box)   # activeFace.box= faceArrays[select_idx,:4]  
                traject['tracking'].reinit()  # Starting from the last filtered obs of detectionTraj              
                          
        elif mode.isInTrackingMode(): 
            faces = faceTracking.track(img)   # No face object yet   TODO rewrite with face objects
            if not faceTracking.isSuccessful or (faceTracking.score < 0.5):      
                mode.switchBackToDetection()   
                traject['tracking'].reinit() 
                continue
            
            observedCenter = faceDetection.returnBoxCenter(faces)  # np.int16        
            traject['tracking'].appendObs(observedCenter)
            traject['tracking'].updateFilter()
      
            if traject['tracking'].needAcquisition():
                traject['tracking'].acquisition(mode.getModeTime())    # TODO: A FAIRE !!?????
            hasToRunRecognition =False
            
        if imgTransfer.isOn : 
                # Faces and video frames are sent to ImgDisplay and with faceRecognitionTask 
                imgTransfer.shareFaces(img, faces, hasToRunRecognition)
                imgTransfer.sendTraject(traject)    # trajectories are sent to image display
          
        
        if ifSaveVideo:   
            fl.saveVideo(video) #TODO ???? VOIR SI CA MARCHE

        if ifSendData:     
            isSent = uart.sendData(activeFace.smoothCenter)   
    cv.destroyAllWindows()
            



# ============================================================================

from sys import platform


def hasPrivileges():
    """
    stackoverflow.com/questions/56177557/detect-os-with-python
    stackoverflow.com/questions/2946746/python-checking-if-a-user-has-administrator-privileges
    """
    if platform == "linux" or platform == "linux2":
        if 'SUDO_USER' in os.environ and os.geteuid() == 0:
            return (os.environ['SUDO_USER'],True)
        else:
            return (os.environ['USERNAME'],False)
        
    elif platform == "Windows":   
        try:
            # only windows users with admin privileges can read the C:\windows\temp
            temp = os.listdir(os.sep.join([os.environ.get('SystemRoot','C:\\windows'),'temp']))
        except:
            return (os.environ['USERNAME'],False)
        else:
            return (os.environ['USERNAME'],True)
