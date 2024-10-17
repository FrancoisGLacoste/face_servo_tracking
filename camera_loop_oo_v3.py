# -*- encoding: utf-8 -*-

import os
from multiprocessing import Queue #, Manager

import numpy as np
import cv2 as cv

from img_transfer import ImgTransfer
from face_detection_yunet_oo_v3 import FaceDetection 
from face_tracking_oo import FaceTracking
from faces import Face 
from trajectory import Trajectory 
from mode import Mode
from uart import UART    # Uses pyserial for serial communication
import file_management as fl    

# ===================================================================             
# Camera loop with face detection, face tracking. 
# It sends coord to servo via serial port, 
# and put face images in a queue for face recognition. 
# ===================================================================             
def cameraLoop(imgTransfer: ImgTransfer):   
    
    if hasPrivileges():
        os.nice(-10)  # High priority task required elevated privileges (sudo)
    
    ifSendData = False
    ifSaveVideo = False 
     
    video = cv.VideoCapture(0)
    faceDetection = FaceDetection(video) 
    faceTracking = FaceTracking()
    
    # Declare trajectories objects for both modes, including the Kalman filters
    trajects = {mode: Trajectory(mode) for mode in ['detection', 'tracking']}
    
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
        
        hasToRunRecognition = False
        if mode.isInDetectionMode(): # TODO: l'objet Mode semble superflu  *********
            traject = trajects['detection']
            faces, largestFaceIndex = faceDetection.detect(img)    #faces: List of Face objects
            if faces is not None:                        
                activeFace = faces[largestFaceIndex]  # a face object    
                traject.appendObs(activeFace.observedCenter)
                if traject.isAtFirstStep(): 
                    traject.filter.setKalmanInitialState(*activeFace.observedCenter)  # 
                traject.updateFilter()
                activeFace.smoothCenter = traject.getLastSmoothPt() 
                
                # Tell the face recognition task if it has to run   
                hasToRunRecognition = False#faceDetection.recognitionCondition(traject['detection'])
                # *****
                
            if  mode.isTimeToSwitchToTracking(faces):
                faceTracking.initTracker(img, activeFace.box)   # activeFace.box= faceArrays[select_idx,:4]  
                traject = trajects['tracking']
                traject.reinit()  # Starting from the last filtered obs of detectionTraj              
                          
        elif mode.isInTrackingMode(): 
            face = faceTracking.track(img)   # Face object, including box and observedCenter in  np.int16  
            if face is None or (face.score < 0.5):   
                # We lost track of the face   
                mode.switchBackToDetection()   
                traject.reinit() 
                continue
                  
            traject.appendObs(face.observedCenter)
            traject.updateFilter()
      
            if traject.needAcquisition():
                traject.acquisition()    # TODO: A FAIRE !!?????
            
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
        return ('SUDO_USER' in os.environ and os.geteuid() == 0)
        '''    return (os.environ['SUDO_USER'],True)
        else:
            return (os.environ['USERNAME'],False)
        '''
        
    '''
    elif platform == "Windows":   
        try:
            # only windows users with admin privileges can read the C:\windows\temp
            temp = os.listdir(os.sep.join([os.environ.get('SystemRoot','C:\\windows'),'temp']))
        except:
            return (os.environ['USERNAME'],False)
        else:
            return (os.environ['USERNAME'],True)
    '''