# -*- encoding: utf-8 -*-


import time, sys
 
import numpy as np
import cv2 as cv

import file_management as fl    # Custom module for file management 
import visual_tracking_oo as vt # Custom module for tracking using vittrack
import filtering as filt     # Custom module for filtering of face center trajectory (Kalman filter)
import uart                  # Custom module using pyserial for serial communication

import visualization as v    
from face_detection_yunet_oo import FaceDetection 
from visual_tracking_oo import FaceTracking
from face_recognition_SFace_oo import FaceRecognition


def weSwitch2Tracking(faces, modeTime, ifSwitch = True):
    """determine when we switch from faceDetection mode to faceTracking mode
    
    modeTime : time.Time(): measures the time spent in detection mode
    """
    if not ifSwitch: 
        return False
    if faces is None: 
        return False
    
    #print(f'detectionModeTime = {modeTime} s')
    # Stay in detection mode for at least 5 sec
    if modeTime < 5:
        return False
    
    print('Switch from detection mode to tracking mode now !')
    return True 
  
# ===================================================================             
def face_servo_tracking(faceRecognition : FaceRecognition):   
    ifSwitch2Tracking = True # False: stay only in detection mode
    ifVisualize = True 
    ifSendData = False
    ifSaveVideo = False 
    ifTrajectAcquisition = False    
        
    # To measure computation times (detection time and frames per second)   
    tm = cv.TickMeter() 
        
    video = cv.VideoCapture(0)

    faceDetection = FaceDetection(video)
    faceDetector = faceDetection.detector
    
    # faceTracking has to maintain some internal states
    faceTracking = FaceTracking()
    faceTracker = faceTracking.tracker
      
    # Initializes UART communication with microcontroller
    if ifSendData: ser = uart.initUART()
    
    select_idx =  None
    
    
    # mode: 'faceDetection' OR 'faceTracking 
    # During faceDetection: if a face is selected, then faceTracking can be activated
    mode = 'faceDetection'
    
    # To measure time spent in face detection mode
    modeStartTime = time.time()
    
    # To follow the trajectory of the selected face center:
    faceCenterTraject = [] # non-filtered observations
    observedTrajectTime = [] # time
         
    
    while video.isOpened():
        isActive, img = video.read()
        img = cv.resize(img, faceDetector.getInputSize())
        
        if not isActive:
            print('Camera not active!')
            break
        
        # Tap any key to exit the loop    
        if cv.waitKey(1) > 0:
            print('Exit the camera loop')
            break
        
        if mode == 'faceDetection': 
            
            tm.reset()    
            tm.start()
            isSuccesful, faces = faceDetector.detect(img) 
            tm.stop()

            # Rem: isSuccesful can be true even when faces is None
            if not isSuccesful or faces is None:
                continue

                    
            if faceRecognition.isActive:
                print('Sending image to face recognition module.')
                
                #Rem: detection output: faces is array[(faceNb,15)]: array of faceNb faces
                # TODO:  devrais mettre cette fct dans une classe ??
                face_imgs = fl.cropBoxes(img, faces[:,:4])   #recognizer.align() ?

                # Put the face image in a (non-async) queue for face recognition
                faceRecognition.putImgQueue(face_imgs)
           
          
            select_idx = faceDetection.selectLargestFace(faces)   # TODO image class
            
            # observedCenter from face detector is noisy and will be filtered in smoothCoord()
            observedCenter = faceDetection.returnFaceCenter(faces, select_idx, 'rectCenter')  # TODO image_box
            
            # TODO trajectory
            faceCenterTraject.append(observedCenter)
            observedTrajectTime.append(time.time())
            
            # TODO smoothCoord avec Kalman 
            
            #if 
            #smoothCenter = filt.smoothCoord(observedCenter)
            smoothCenter = observedCenter
            
            if ifVisualize:   
                # TODO: re-organize the visualization modules AND the trajectory filtering modules
                v.visualizeTraject_inDetection(faceDetection,
                                               img, faces, smoothCenter,select_idx, tm, 
                                               faceCenterTraject  )
           
            if  weSwitch2Tracking(faces, time.time() - modeStartTime, ifSwitch2Tracking) : 
                assert faces is not None
                face_to_track  = np.round(faces[select_idx,:4]).astype(np.int16) # [x,y,w,h]            
                
                faceTracking.initTracker(img, face_to_track)
                
                if ifSaveVideo:   
                    fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                                
                mode = 'faceTracking'
                lastCenter = smoothCenter #last filter prediction from detection
                
                # Initialize Kalman Filter for face tracking     
                x0, y0 = smoothCenter[:2]
                trackingFilter,filteredTraject=filt.initFiltering((x0, y0),
                                                            'tracking')
                modeStartTime = time.time() # to measure the time in faceTracking mode
                faceCenterTraject   = [] # list of measurements/observations of face centers
                observedTrajectTime = []

                            
        elif mode == 'faceTracking':
            """  The faceTracking mode can only be access from the faceDetection mode 
            Both faceTracking mode and faceDetection mode send face coordinates 
            to the microcontroller for servo-tracking.
            """
            
            tm.reset()
            tm.start()
            #isLocated, bbox, score = faceTracker.infer(img) # TODO
            isLocated, faceTuple = faceTracker.update(img)
            score = faceTracker.getTrackingScore()
            tm.stop()
            
            if not isLocated or (score < 0.5):      
                # Switch back to face detection mode
                print(f'''Target is lost: go back to face detection after 
                      spending {modeStartTime - time.time()} s in face tracking mode.''')
                mode ='faceDetection'
                modeStartTime = time.time() # now measures the time in faceDetection mode
                continue
            
            observedCenter_int16 = faceDetection.returnBoxCenter(faceTuple)  # np.int16
            #print(observedCenter_int16)
            observedCenter = np.array(observedCenter_int16, np.float32).reshape(2,1) # TODO: rester consistent dans le choix de type (float32 vs int16 etc)
            faceCenterTraject.append(observedCenter) #i.e. measurements.append(...)
            observedTrajectTime.append(time.time())
 
            #print(observedCenter.shape)  #(2,1)
            
            #Kalman Filtering is used to smooth the observed trajectory of face centers
            filteredTraject = filt.updateFiltering(observedCenter,trackingFilter, 
                                                   filteredTraject)
            smoothCenter = filteredTraject[-1] 

            if ifVisualize:
                v.visualizeTraject_inTracking( img,faceTuple, smoothCenter,select_idx, tm, 
                                                faceCenterTraject, filteredTraject)    
                if ifSaveVideo:
                    fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                    continue 
          
        if ifTrajectAcquisition:  
            acquisitionTime = 60 #[s]   1 min        
            if time.time() - modeStartTime > acquisitionTime:
                # Signal acquisition for state model estimation
          
                # TODO:  pass an object Trajectory as argument
                fl.saveTraject(faceCenterTraject, observedTrajectTime, mode)
                # Reset the face center trajectory
                faceCenterTraject   = [] # Non-filtered observations
                observedTrajectTime = [] # 
        
            
        if ifSendData: 
            isSent = uart.sendData(ser, smoothCenter)
           
    cv.destroyAllWindows()
            

# ========================  Test ============ ======================================
def test_face_servo_tracking_1(faceRecognition):
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

   