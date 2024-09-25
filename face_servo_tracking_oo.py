# -*- encoding: utf-8 -*-


import time
 
import numpy as np
import cv2 as cv

# Custom modules and classes
from face_detection_yunet_oo import FaceDetection 
from visual_tracking_oo import FaceTracking
from face_recognition_SFace_oo import FaceRecognition
from trajectory import Trajectory
#import visualization
from image import Image
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
    imgDisplay = Image()
    
    faceDetection = FaceDetection(video)
    faceDetector = faceDetection.detector
    detectionTraject = Trajectory('detection') # including a Kalman filter
    
    faceTracking = FaceTracking()
    faceTracker = faceTracking.tracker
    trackingTraject = Trajectory('tracking')  # including a Kalman filter
      
    # Initializes UART communication with microcontroller
    if ifSendData: ser = uart.initUART()
    
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
            
            img = cv.resize(img, faceDetector.getInputSize())
            
            imgDisplay.setNewFrame(img)
            
            
            tm.reset()    
            tm.start()  # TODO :  can we have processingTime as member of FaceDetector ?
            isSuccesful, faces = faceDetector.detect(img) 
            #print(type(faces)) # array(1,15)
            tm.stop()
           
            
            if isSuccesful and faces is not None:  
                faceDetection._incrementStep()
                select_idx = faceDetection.selectLargestFace(faces)   # TODO image class
                observedCenter = faceDetection.returnFaceCenter(faces, select_idx, 'rectCenter')  # TODO image_box
                    
                detectionTraject.appendObs(observedCenter)
                if detectionTraject.isAtFirstStep(): 
                    detectionTraject.filter.setKalmanInitialState(*observedCenter)
                detectionTraject.updateFilter()
                
                imgDisplay.visualize(detectionTraject, faces, tm, None, select_idx )
            

                #------ ??? DEVRAIS-JE CREER UNE NOUVELLE CLASSE 'Face' (ou kekchose du genre) 
                # On veut comparer la courante face avec la precedente face: centre, select_idx
                # Parfois j'ai [francois, francois, unrecognized, francois,...]
                # Ou pire: [francois, francois, audrey, francois,...]
                # On veut se baser sur la continuite des centres des images pour conclure 
                # a la continuite des faceName 
                # (c-a-d que ci-haut, audrey et unrecognized devraient etre francois)
                # Mais ca s'infere seulement a posteriori. On peut pas deviner sur le champs !
                nearPreviousFace  = (detectionTraject.distance() < 6)  
                likelyTheSameFace = bool(  np.mod(faceDetection.step+1,5 )) and nearPreviousFace  

                # Le probleme sera aussi de detecter un saut d'une face a une autre quand il y 
                # en a plus d'une.... 
                # Et on voudra ne permettre le saut qu'a certaines conditions. 
                
                condition = faceRecognition.isActive()              \
                    and (detectionStep ==1 or not likelyTheSameFace )   \
                    and not detectionTraject.inFastMotion()        
                #------------      
                                
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
            
            trackingTraject.appendObs(observedCenter)
            trackingTraject.updateFilter()
            
            
            imgDisplay.visualize(trackingTraject, faces, tm, score )      
        
            if ifSaveVideo:
                fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                continue 
    
        
        if imgDisplay.ifVisualize: 
                imgDisplay.show()#cv.imshow('Video', imgDisplay)
        
        # TODO UTILE ou non ??
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

   