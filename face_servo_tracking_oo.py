# -*- encoding: utf-8 -*-


import time, sys
 
import numpy as np
import cv2 as cv

import file_management as fl    # Custom module for file management 
import visual_tracking_oo as vt # Custom module for tracking using vittrack

import uart                  # Custom module using pyserial for serial communication
import visualization as v    
from face_detection_yunet_oo import FaceDetection 
from visual_tracking_oo import FaceTracking
from face_recognition_SFace_oo import FaceRecognition
from trajectory import Trajectory
import filtering as filt


class Mode:
    # Constant parameters: 
    minTimeInDetectionMode = 5 # sec

    def __init__(self):
        self._modeTime = 0               # initialize mode time to 0
        self._mode = 'faceDetection'     # initialize the mode to face detection
    
        # To measure time spent in each mode
        self.startTime = time.time()
 
    def isTimeToSwitchToTracking(self, faces):
        """determine when the mode switches from faceDetection mode to faceTracking mode
        """
        if faces is None: 
            return False
        
        # Stay in detection mode for at least {minTimeInDetectionMode} sec
        if self._modeTime < self.minTimeInDetectionMode:
            return False
        
        else:
            self._mode = 'faceTracking'
            print('Switch from detection mode to tracking mode now !')
            self.startTime = time.time() # to measure the time in faceTracking mode
            return True 
  
    def switchBackToDetection(self):
        # Switch back to face detection mode
        print(f'''Target is lost: go back to face detection after 
                spending {self.getModeTime()} s in face tracking mode.''')
        self._mode ='faceDetection'
        self.startTime = time.time() # to measure the time in faceDetection mode
        
    def get(self):
        return self._mode  
    
    def getModeTime(self):
        return self.startTime - time.time()  
    
    def inDetectionMode(self):
        return self.get() == 'faceDetection'
    
    def inTrackingMode(self):
        return self.get() == 'faceTracking'
    
# ===================================================================             
# Camera loop with face detection, face tracking. 
# It sends coord to servo via serial port, 
# and put face images in a queue for face recognition. 
# ===================================================================             
def face_servo_tracking(faceRecognition : FaceRecognition):   
    
    ifVisualize = True 
    ifSendData = False
    ifSaveVideo = False 
    ifTrajectAcquisition = False    
        
    # To measure computation times (detection time and frames per second)   
    tm = cv.TickMeter() 
        
    video = cv.VideoCapture(0)

    faceDetection = FaceDetection(video)
    faceDetector = faceDetection.detector
    detectionTraject = Trajectory()
    
    # faceTracking has to maintain some internal states
    faceTracking = FaceTracking()
    faceTracker = faceTracking.tracker
    trackingTraject = Trajectory()
      
    # Initializes UART communication with microcontroller
    if ifSendData: ser = uart.initUART()
    
    select_idx =  None
    
    
    # mode: 'faceDetection' OR 'faceTracking 
    # During faceDetection: if a face is selected, then faceTracking can be activated
    mode = Mode()
    
   
    # To follow the trajectory of the selected face center:
    #faceCenterTraject = [] # non-filtered observations
    #observedTrajectTime = [] # time
    detectionTraject.reinit()     
    
    while video.isOpened():
        isActive, img = video.read()
        
        
        if not isActive:
            print('Camera not active!')
            break
        
        # Tap any key to exit the loop    
        if cv.waitKey(1) > 0:
            print('Exit the camera loop')
            break
        
        if mode.inDetectionMode(): 
            img = cv.resize(img, faceDetector.getInputSize())
            
            tm.reset()    
            tm.start()  # TODO :  can we have processingTime as member of FaceDetector ?
            isSuccesful, faces = faceDetector.detect(img) 
            tm.stop()

            # Rem: isSuccesful can be true even when faces is None        
            if isSuccesful and faces is not None:  
                                
                if faceRecognition.isActive() and not detectionTraject.inFastMotion():
                    print('Sending image to face recognition module.')
                    
                    #Rem: detection output: faces is array[(faceNb,15)]: array of faceNb faces
                    # TODO:  devrais mettre cette fct dans une classe ??
                    face_imgs = fl.cropBoxes(img, faces[:,:4])   #recognizer.align() ?

                    # Put the face image in a (non-async) queue for face recognition
                    faceRecognition.putImgQueue(face_imgs)
            
            
                select_idx = faceDetection.selectLargestFace(faces)   # TODO image class
                
                # observedCenter from face detector is noisy and will be filtered in smoothCoord()
                observedCenter = faceDetection.returnFaceCenter(faces, select_idx, 'rectCenter')  # TODO image_box
                
                detectionTraject.append(observedCenter)
                smoothCenter = observedCenter
                
                
                # TODO: re-organize the visualization modules AND the trajectory filtering modules
                img =v.visualizeTraject_inDetection(faceDetection, detectionTraject,
                                                img, faces,select_idx, tm  )
        
            if  mode.isTimeToSwitchToTracking(faces): 
                
                face_to_track  = np.round(faces[select_idx,:4]).astype(np.int16) # [x,y,w,h]            
                faceTracking.initTracker(img, face_to_track)
                
                if ifSaveVideo:   
                    fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                
                trackingTraject.reinit()                
                
                #lastCenter = smoothCenter #last filter prediction from detection
                #x0, y0 = smoothCenter[:2]
                #trackingFilter,filteredTraject=filt.initFiltering((x0, y0), 'tracking')
                

                            
        elif mode.inTrackingMode(): 
            """  The faceTracking mode can only be access from the faceDetection mode 
            """
            
            tm.reset() # to monitor tracking algo performance 
            tm.start()
            #isLocated, bbox, score = faceTracker.infer(img) # TODO
            isSuccessful, faceTuple = faceTracker.update(img)
            score = faceTracker.getTrackingScore()
            tm.stop()
            
            if not isSuccessful or (score < 0.5):      
                mode.switchBackToDetection()    
                continue
            
            observedCenter = faceDetection.returnBoxCenter(faceTuple)  # np.int16
            # int16 has to be transformed into float32 for the kalman filter to work on it        
            
            trackingTraject.update(observedCenter)
            '''
            observedCenter = np.array(observedCenter_int16, np.float32).reshape(2,1) # TODO: rester consistent dans le choix de type (float32 vs int16 etc)
            faceCenterTraject.append(observedCenter) #i.e. measurements.append(...)
            observedTrajectTime.append(time.time())
 
            #print(observedCenter.shape)  #(2,1)
            
            
            #Kalman Filtering is used to smooth the observed trajectory of face centers
            filteredTraject = filt.updateFiltering(observedCenter,trackingFilter, 
                                                   filteredTraject)
            smoothCenter = filteredTraject[-1] 

            '''            
            # TODO: reorganize the all visualization thing.
            img = v.visualizeTraject_inTracking( faceTracking, trackingTraject, 
                                                img,faceTuple,score, tm)    
            if ifSaveVideo:
                fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                continue 
    
        
        if ifVisualize: 
                cv.imshow('Video', img)
                
        if ifTrajectAcquisition:  
            acquisitionTime = 60 #[s]   1 min        
            if time.time() - mode.startTime > acquisitionTime:
                # Signal acquisition for state model estimation
      
                # TODO pour a la fois detection que pour tracking
                fl.saveTraject(detectionTraject, mode)         
                detectionTraject.reinit()       
            
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

   