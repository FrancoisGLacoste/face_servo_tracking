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
            
            print(img.shape)
            img = cv.resize(img, faceDetection.detector.getInputSize())
            print(img.shape)   # (576, 768, 3)
            imgDisplay.setNewFrame(img)
            faces = faceDetection.detect(img)
            
            if faceDetection.isSuccessful and faces is not None:  
                faceDetection._incrementStep()   # TODO: hid it , but where ?
                select_idx = faceDetection.selectLargestFace(faces)   # TODO ? Face class
                observedCenter = faceDetection.returnFaceCenter(faces, select_idx, 'rectCenter')  # TODO image_box
                    
                detectionTraject.appendObs(observedCenter)
                if detectionTraject.isAtFirstStep(): 
                    detectionTraject.filter.setKalmanInitialState(*observedCenter)
                detectionTraject.updateFilter()
                
                tm = faceDetection.tm
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
                
                condition =False 
                ''' faceRecognition.isActive()              \
                    and (detectionStep ==1 or not likelyTheSameFace )   \
                    and not detectionTraject.inFastMotion()        
                #------------    '''  
                                
                if condition:
                    print('Sending image to face recognition module.')
                    
                    #Rem: detection output: faces is array[(faceNb,15)]: array of faceNb faces
                    # TODO:  devrais mettre cette fct dans une classe ??
                    face_imgs = fl.cropBoxes(img, faces[:,:4])   
                    '''recognizer.alignCrop(src_img: cv2.typing.MatLike, face_box: cv2.typing.MatLike, aligned_img: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
                        
                        recognizer.alignCrop( src_img: UMat, face_box: UMat, aligned_img: UMat | None = ...) -> UMat: ...
                    '''
                    # Put the face image in a (non-async) queue for face recognition
                    faceRecognition.putImgQueue(face_imgs)
                
            

            if  mode.isTimeToSwitchToTracking(faces): 
                print(img.shape)
                faceTracking.initTracker(img, faces[select_idx,:4])                
                print(img.shape)
                trackingTraject.reinit()  # Starting from the last filtered obs of detectionTraj              
                          
        elif mode.isInTrackingMode(): 
            #print(img.shape)   # (480, 640, 3)
            imgDisplay.setNewFrame(img)
            faces = faceTracking.track(img)
            if not faceTracking.isSuccessful or (faceTracking.score < 0.5):      
                mode.switchBackToDetection()    
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

   