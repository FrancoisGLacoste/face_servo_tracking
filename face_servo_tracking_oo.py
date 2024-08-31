# -*- encoding: utf-8 -*-
 
import numpy as np
import cv2 as cv

import sys, os, time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import file as fl            # Custom module for file management 
#import face_detection_yunet as dtc # Custom module for face detection using yunet
import visual_tracking_oo as vt # Custom module for tracking using vittrack
import filtering as filt     # Custom module for filtering of face center trajectory (Kalman filter)
import uart                  # Custom module using pyserial for serial communication
#import face_recognition_SFace_PAS_FINI as recog  # custom module for face recognition 


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
def face_servo_tracking(faceRecognition):   
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
            break
        
        
        if mode == 'faceDetection': 
            
            tm.reset()    
            tm.start()
            isSuccesful, faces = faceDetector.detect(img) 
            tm.stop()
            
            #print(faces.shape)
            
            # Rem: isSuccesful can be true even when faces is None
            if not isSuccesful:
                continue

            if faces is None: 
                continue
                    
            if faceRecognition.isActive:
                print('Sending image to face recognition module.')
                
                #Rem: detection output: faces is array[(faceNb,15)]: array of faceNb faces
                # TODO:  devrais mettre cette fct dans une classe ??
                face_imgs = fl.cropBoxes(img, faces[:,:4])   #recognizer.align() ?

                # Put the face image in the queue for face recognition
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
            
            if ifVisualize:    # TODO image class
                # Rem: output img can be written into a video (videoWrite) 
                img = faceDetection.visualizeDetection(img, faces, smoothCenter,select_idx, tm)
                img = filt.visualizeTraject(img, faceCenterTraject)
                cv.imshow('Video', img)
            
           
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
 
            #print(observedCenter)
            #print(observedCenter.shape)  #(2,1)
            
            #Kalman Filtering is used to smooth the observed trajectory of face centers
            filteredTraject = filt.updateFiltering(observedCenter,trackingFilter, 
                                                   filteredTraject)
            smoothCenter = filteredTraject[-1] 

            #print(smoothCenter.shape) # (4,1)
            #print(type(faceCenterTraject))
            #print(type(faceCenterTraject[0].shape))  # (2,1)
            

            if ifVisualize:
                GREEN = (10,255,0)
                BLUE = (255,0,0)
                img = vt.visualizeTracking(img, faceTuple, smoothCenter[:2], score, tm)
                img = filt.visualizeTraject(img, faceCenterTraject, GREEN)
                img = filt.visualizeTraject(img, filteredTraject, BLUE)
                cv.imshow('Video', img)
                
                if ifSaveVideo:
                    fl.saveVideo(video) #TODO  VOIR SI CA MARCHE
                    continue 
          
        if ifTrajectAcquisition:  
            acquisitionTime = 60 #[s]   1 min        
            if time.time() - modeStartTime > acquisitionTime:
                # Signal acquisition for state model estimation
          
                filt.saveTraject(faceCenterTraject, observedTrajectTime, mode)
                # Reset the face center trajectory
                faceCenterTraject   = [] # Non-filtered observations
                observedTrajectTime = [] # 
        
            
        if ifSendData: 
            isSent = uart.sendData(ser, smoothCenter)
           
    cv.destroyAllWindows()
            
async def main():   
    
    # Create a FaceRecognition object, including input/result queues
    faceRecognition = FaceRecognition(isActive = True) 
    faceRecognition.prepareFaceRecognition() # compute features, train the kNN classifier, save that.
       
    # Create the (CPU-bound) process for the servo-tracking of faces
    camera_process = mp.Process(target=face_servo_tracking, 
                           args=(faceRecognition,) )
    camera_process.start()
     
    # During the face servo-tracking, face recognition is performed on a separate thread pool
    await asyncio.gather(
            faceRecognition.runFaceRecognitionTask(),
            faceRecognition.retrieveResults()  # and results are retrieved 
            )
    
    camera_process.join()
    
    
# ========================  Test 1 ======================================
def test_face_servo_tracking_1(faceRecognition):
    while True:
        # e.g.  Face Detection
        mp.Event().wait(0.1)  #  delay    
        newData = np.random.randint(0, 100)
        print(f'Put {newData} in  inputQueue')
        faceRecognition.putImgQueue(newData)


         
async def test_retrieve_results_1(resultQueue):
    print("This is in retrieve_function ")
    while True:
        result = await resultQueue.get()
        print(f"Result received: {result}")
        resultQueue.task_done()

async def test_main_async_process():
    """ Test 1 
        To test the asyncio/process/threads structure
        Without doing anything but passing numbers as data
    test_face_servo_tracking_1 <----- face_servo_tracking  
    test_retrieve_results_1    <----- retrieve_results
    faceRecognition.test_runTask 
    """
    faceRecognition = FaceRecognition() 
    camera_process = mp.Process(target=test_face_servo_tracking_1, 
                           args=(faceRecognition,))
    camera_process.start()
    await asyncio.gather(
            faceRecognition.test_runTask(),
            test_retrieve_results_1(faceRecognition.resultQueue)       
            )
    
    camera_process.join()    
         
if __name__ == '__main__':
    #asyncio.run(test_main_async_process())
 
    asyncio.run(main())
    