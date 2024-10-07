# -*- encoding: utf-8 -*-

import os
from multiprocessing import Queue #, Manager

import numpy as np
import cv2 as cv

# Custom modules and classes
from img_transfer import ImgTransfer
from face_detection_yunet_oo_v3 import FaceDetection 
from face_tracking_oo import FaceTracking
#from face_recognition_SFace_oo_v3 import FaceRecognition
from trajectory import Trajectory
#from image_display_v3 import ImageDisplay
from mode import Mode
from uart import UART    # Uses pyserial for serial communication
import visualization_v3 as v    
import file_management as fl    

# ===================================================================             
# Camera loop with face detection, face tracking. 
# It sends coord to servo via serial port, 
# and put face images in a queue for face recognition. 
# ===================================================================             
def cameraLoop(imgTransfer: ImgTransfer, resultQueue: Queue):   
    
    #os.nice(-10)  # High priority task required elevated privileges (sudo)
    
    ifSendData = False
    ifSaveVideo = False 
     
    video = cv.VideoCapture(0)
    #imgDisplay = ImageDisplay()                                    # To display video frames
    faceDetection = FaceDetection(video) 
    faceTracking = FaceTracking()
    
    # Declare trajectories objects for both modes, including the Kalman filters
    traject = {mode: Trajectory(mode) for mode in ['detection', 'tracking']}
    
    imgTransfer.createSharedMemory(faceDetection.frameSize) # shared_memory for image transfer 
    
    if ifSendData: uart = UART()               # Serial communication with microcontroller
    
    select_idx =  None
    
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
            #'''imgDisplay.setNewFrame(img)'''
            faces = faceDetection.detect(img)
            
            if faceDetection.isSuccessful and faces is not None:  
                
                select_idx = faceDetection.selectLargestFace(faces)   # TODO ? Face class
                observedCenter = faceDetection.returnFaceCenter(faces, select_idx, 'rectCenter')  # TODO image_box
                    
                traject['detection'].appendObs(observedCenter)
                if traject['detection'].isAtFirstStep(): 
                    traject['detection'].filter.setKalmanInitialState(*observedCenter)
                traject['detection'].updateFilter()
                
                # Tell the face recognition task if it has to run
                hasToRunRecognition = faceDetection.recognitionCondition(traject['detection'])
        
          
            if  mode.isTimeToSwitchToTracking(faces): #(480, 640, 3)
                faceTracking.initTracker(img, faces[select_idx,:4])                
                traject['tracking'].reinit()  # Starting from the last filtered obs of detectionTraj              
                          
        elif mode.isInTrackingMode(): # (480, 640, 3)
            faces = faceTracking.track(img)
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
                # Share the video frames with ImgDisplay and with faceRecognitionTask 
                imgTransfer.share(img, faces, hasToRunRecognition)
                
                #'''imgDisplay.setNewFrame(img)'''    
                # TODO: "visualization will be on a Tornado server
                ''' imgDisplay.visualize(transferQueue, traject['detection'], faces, faceDetection.tm, 
                                     None, select_idx )
                '''             
                '''
                    imgDisplay.visualize(resultQueue, traject['tracking'], 
                                    faces, faceTracking.tm, faceTracking.score )      
                '''
        
        if ifSaveVideo:   
            fl.saveVideo(video) #TODO ???? VOIR SI CA MARCHE

        if ifSendData: 
            # traject[mode.get()]: trajectory object of the active mode (detection or tracking)
            lastSmoothObs = traject[mode.get()].smoothObs[-1][:2]
            isSent = uart.sendData(lastSmoothObs)   
    cv.destroyAllWindows()
            
