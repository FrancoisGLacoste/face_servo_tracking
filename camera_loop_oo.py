# -*- encoding: utf-8 -*-

import numpy as np
import cv2 as cv

# Custom modules and classes
from face_detection_yunet_oo import FaceDetection 
from face_tracking_oo import FaceTracking
from face_recognition_SFace_oo import FaceRecognition
from trajectory import Trajectory
from image import Image
from mode import Mode
from uart import UART    # Uses pyserial for serial communication
import visualization as v    
import file_management as fl    

# ===================================================================             
# Camera loop with face detection, face tracking. 
# It sends coord to servo via serial port, 
# and put face images in a queue for face recognition. 
# ===================================================================             
def camera_loop(faceRecognition : FaceRecognition):   
    
    #ifVisualize = True 
    ifSendData = False
    ifSaveVideo = False 
     
    video = cv.VideoCapture(0)
    imgDisplay = Image()
    
    faceDetection = FaceDetection(video)
    detectionTraject = Trajectory('detection') # including a Kalman filter
    faceTracking = FaceTracking()
    trackingTraject = Trajectory('tracking')  # including a Kalman filter
      
    if ifSendData: uart = UART() # Serial communication with microcontroller
    
    select_idx =  None
    detectionStep =0
    
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
            
            #print(img.shape) # (480, 640, 3)
            #img = cv.resize(img, faceDetection.detector.getInputSize())
            #print(img.shape)   # (480, 640, 3)   , (576, 768, 3)
            imgDisplay.setNewFrame(img)
            faces = faceDetection.detect(img)
            
            if faceDetection.isSuccessful and faces is not None:  
                
                select_idx = faceDetection.selectLargestFace(faces)   # TODO ? Face class
                observedCenter = faceDetection.returnFaceCenter(faces, select_idx, 'rectCenter')  # TODO image_box
                    
                detectionTraject.appendObs(observedCenter)
                if detectionTraject.isAtFirstStep(): 
                    detectionTraject.filter.setKalmanInitialState(*observedCenter)
                detectionTraject.updateFilter()
                
                # TODO: "visualize" functions still require refactoring ***********
                imgDisplay.visualize(detectionTraject, faces, faceDetection.tm, None, select_idx )
                             
                if faceDetection.recognitionCondition(detectionTraject,faceRecognition):
                    faceRecognition.sendToFaceRecognition(img, faces[:,:4])
                
            

            if  mode.isTimeToSwitchToTracking(faces): 
                #print(img.shape) #(480, 640, 3)
                faceTracking.initTracker(img, faces[select_idx,:4])                
                trackingTraject.reinit()  # Starting from the last filtered obs of detectionTraj              
                          
        elif mode.isInTrackingMode(): 
            #print(img.shape)   # (480, 640, 3)
            imgDisplay.setNewFrame(img)
            faces = faceTracking.track(img)
            if not faceTracking.isSuccessful or (faceTracking.score < 0.5):      
                mode.switchBackToDetection()   
                trackingTraject.reinit() 
                continue
            
            observedCenter = faceDetection.returnBoxCenter(faces)  # np.int16        
            trackingTraject.appendObs(observedCenter)
            trackingTraject.updateFilter()
            
            imgDisplay.visualize(trackingTraject, faces, faceTracking.tm, faceTracking.score )      
         
            if trackingTraject.needAcquisition():
                trackingTraject.acquisition(mode.getModeTime())    # TODO: A FAIRE !!
        
        if imgDisplay.ifVisualize: 
                imgDisplay.show()#cv.imshow('Video', imgDisplay)

        if ifSaveVideo:   
            fl.saveVideo(video) #TODO  VOIR SI CA MARCHE

        if ifSendData: 
            isSent = uart.sendData(Trajectory.lastSmoothPt[:2])   
    cv.destroyAllWindows()
            
# ========================  Test ============ ======================================
def test_camera_loop_1(faceRecognition):
    while True:
        # e.g.  Face Detection
        #  Compute something silly for the pure sake of wasting time (i.e. to delay...)     
        newData = np.mean(np.random.randint(0, 1000))
        print(f'Put {newData} in  inputQueue')
        faceRecognition.putImgQueue(newData)


         
async def test_retrieve_results_1(resultQueue):
    print("This is in retrieve_function ")
    while True:
        result = await resultQueue.get()
        print(f"Result received: {result}")
        resultQueue.task_done()

   