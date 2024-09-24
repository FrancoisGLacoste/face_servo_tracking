# -*- encoding: utf-8 -*-


import time
 
import numpy as np
import cv2 as cv

# Custom modules and classes
from face_detection_yunet_oo import FaceDetection 
from visual_tracking_oo import FaceTracking
from face_recognition_SFace_oo import FaceRecognition
from trajectory import Trajectory
from mode import Mode
import uart                     # Custom module using pyserial for serial communication
import visualization as v    
import file_management as fl    

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
    detectionTraject = Trajectory('detection') # including a Kalman filter
    
    faceTracking = FaceTracking()
    faceTracker = faceTracking.tracker
    trackingTraject = Trajectory('tracking')  # including a Kalman filter
      
    # Initializes UART communication with microcontroller
    if ifSendData: ser = uart.initUART()
    
    select_idx =  None
    faceIndex =0
    
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
            
            img = cv.resize(img, faceDetector.getInputSize())
            
            tm.reset()    
            tm.start()  # TODO :  can we have processingTime as member of FaceDetector ?
            isSuccesful, faces = faceDetector.detect(img) 
            #print(type(faces)) # array(1,15)
            tm.stop()

            # Rem: isSuccesful can be true even when faces is None        
            if isSuccesful and faces is not None:  
                   
          
                select_idx = faceDetection.selectLargestFace(faces)   # TODO image class
                observedCenter = faceDetection.returnFaceCenter(faces, select_idx, 'rectCenter')  # TODO image_box
                    
                detectionTraject.appendObs(observedCenter)
                if detectionTraject.isAtFirstStep(): 
                    detectionTraject.filter.setKalmanInitialState(*observedCenter)
                detectionTraject.updateFilter()
                
                # TODO: re-organize the visualization modules AND the trajectory filtering modules
                img =v.visualizeTraject_inDetection(detectionTraject.observations, 
                                                    detectionTraject.filteredObs,
                                                img, faces,select_idx, tm  )

                #Compter les images de face capturees: 
                faceIndex =+1
                # On veut comparer la courante face avec la precedente face: centre, select_idx
                # Etudier la continuite des centres et des faceName dans la trajectoire. 
                nearPreviousFace  = (detectionTraject.distance() < 6)  
                likelyTheSameFace = bool(  np.mod(faceIndex+1,5 )) and nearPreviousFace  
            
                #if previousFace.name
                #print('l2 distance:', detectionTraject.distance())
                # Le probleme sera davantage de detecter un saut d'une face a une autre.
                
                condition = faceRecognition.isActive()              \
                    and (faceIndex ==1 or not likelyTheSameFace )   \
                    and not detectionTraject.inFastMotion()        
                      
                                
                if condition:
                    print('Sending image to face recognition module.')
                    
                    #Rem: detection output: faces is array[(faceNb,15)]: array of faceNb faces
                    # TODO:  devrais mettre cette fct dans une classe ??
                    face_imgs = fl.cropBoxes(img, faces[:,:4])   #recognizer.align() ?

                    # Put the face image in a (non-async) queue for face recognition
                    faceRecognition.putImgQueue(face_imgs)
                
            

            if  mode.isTimeToSwitchToTracking(faces): 
                
                face_to_track  = np.round(faces[select_idx,:4]).astype(np.int16) # [x,y,w,h]            
                faceTracking.initTracker(img, face_to_track)
                
                #if ifSaveVideo:   
                #    fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                
                trackingTraject.reinit()  # Starting from the last filtered obs of detectionTraj              
                          
        elif mode.isInTrackingMode(): 
            
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
            img = v.visualizeTraject_inTracking(trackingTraject.observations,
                                                trackingTraject.filteredObs, 
                                                img,faceTuple,score, tm)    
            if ifSaveVideo:
                fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                continue 
    
        
        if ifVisualize: 
                cv.imshow('Video', img)
        
        # TODO ??
        '''        
        if ifTrajectAcquisition:  
            acquisitionTime = 60 #[s]   1 min        
            if time.time() - mode.startTime > acquisitionTime:
                # Signal acquisition for state model estimation
      
                # TODO pour a la fois detection que pour tracking
                fl.saveTraject(detectionTraject, mode)         
                detectionTraject.reinit()       
        '''    
        
        if ifSendData: 
            isSent = uart.sendData(ser, Trajectory.lastSmoothPt[:2])
           
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

   