# -*- encoding: utf-8 -*-

"""
face_recognition_SFace_oo_v3.py    version with shared_memory
"""
import os, time 
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue as qu

#To transfer image: either mp.queue or mp.shared_memory 
import multiprocessing as mp

import cv2 as cv
import numpy as np

from img_transfer import ImgTransfer
import new_face_gui_tk as gui
from sface_embeddings import SFaceEmbeddings
from knn_classifier import KnnClassifier
from unrecognition_criteria import UnrecognitionCriteria
import file_management as fl
import util as u

"""  ====================== SFACE ======================================= 
SFace : [Zhonyu2021] a state-of-the-art algorithm for face recognition
Ref:
https://github.com/zhongyy/SFace/blob/main/SFace_torch/train_SFace_torch.py
https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py#L99

======================================================================="""
# ================== For the execution of face recognition loop ===============          
def faceRecognitionLoop(imgTransfer: ImgTransfer, resultQueue: mp.Queue):
    """ 
    """
    # Create a FaceRecognition object, including input/result queues
    faceRecognition = FaceRecognition() 
    print('We just create the faceRecognition object in faceRecognitionLoop')
    
    while True:
        # Receive a face image from the face detection loop
        print('We are receiving the last face image that has been detected.')
        img, boxes =imgTransfer.retrieveImage()   
        print('We got a new face image for the face recognition task to process.')
        
        # Face recognition per se 
        faceImgs = faceRecognition.cropBoxes(img, boxes)    
        results=list()
        for faceImg in faceImgs:            
            recognizedName, certainty = faceRecognition.recognizeFace(faceImg)
            results.append((faceImg, recognizedName, certainty))
            index = '?'
            resultQueue.put([index, recognizedName, certainty])
            # Transfer the results to imgDisplay via a mp.queue or via mp.Manager
            # TODO How to synchronize the display of recognizedName and certainty on the video frame ?
            # ? Est-ce je devrais pas numeroter les frames (un index) pour nous assurer de 
            # retourner les resultats a la bonne image ?
        
        '''# Put the result in the result queue: the result is sent to retrieveResults()
        await self.resultQueue.put(results) 
        #TODO For now it is sent via a queue, but it should 
        # eventually be connected via TCP/IP to a remote client that runs a GUI for the user.  
        '''
        
class FaceRecognition:
    
    def __init__(self , isActive = True, faceNames= None):
        self._isActive = isActive  
        self.faceNames = self.returnFaceNames(faceNames) # face names we want to recognize
         
         
        #self.resultQueue # voir imgTransfer()
        #asyncio.Queue() # Queue for capturing the recognition results (i.e. face names)
                    
        if self._isActive:
            self.sFace = SFaceEmbeddings(self.faceNames)  # SFace features Embeddings
            self.sFace.stackFaceEmbeddingsInArray()       # set self.sFace.X, self.ace.sFy
            self.knn = KnnClassifier(self.sFace.X, self.sFace.y, self.faceNames)
            self.unrecognitionCriteria = UnrecognitionCriteria(self.sFace,self.faceNames)    
             
            # ------------------ For new face identification  --------------------------------------
            self.newFaceIdThread = ThreadPoolExecutor(max_workers=1) # Thread for running newFaceId 
            # Queue for moving a new face identity (faceName) from  newFaceIdGUI
            self.newFaceIdQueue = qu.Queue() 
            self.newFaceIdGUI = None # Only created when required for new face identification
            self.idTime = 0
        
            print('FaceRecognition preparation has been done.')  
        else: 
            print('Face recognition is inactive: no SFace embeddings, no kNN: Nada! нет!')
        
    def afterNewFaceId(self) : 
        # 
        timeAfterNewFaceId =  time.time()  -  self.idTime
        if timeAfterNewFaceId < 120 : # 1 minute
            return True
              
    def isActive(self ):
        
        # When a new face has just been id: dont ask for recognition again for some time    
        if not self.afterNewFaceId():
            return self._isActive  
        
         
    def cropBoxes(self,img, boxes, inGray=False):
        # Likely to be replaced by recognizer.alignCrop()  
        """ 
        Arg: 
            img:   np.ndarray         a single image
            boxes: np.ndarray([:,:4]) or np.ndarray([:,:15]) : array of boxes, i.e. coords (x,y,w,h), e.g of the faces.
        Returns:    list of  images contained by the boxes, (gray if asked)    
        """
        #print(boxes[0]) # valid both for lists and arrays
        if boxes is None or len(boxes.squeeze())==0: 
            print('No box to crop from the image')
            return []
        if inGray : 
            img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
        croppedImgs =[self.sFace.recognizer.alignCrop(img, box) for box in boxes ]
        croppedImgs_2 =[img[y:y+h,x:x+w] for (x,y,w,h) in boxes[:,:4].astype(int)]
        return croppedImgs
    
    # =============================================================================        
             
            
    async def retrieveResults(self): # send results to GUI when unrecognized
        """ Retrieves results from face recognition of the face images 
            from a video frame.
           The unrecognized face are identified by the user, and saved as {name}_{count+1}.jpeg 
           the new directory {name} 
        """
        
        # TODO: we want to send the result via TCP/IP client-server communication
        # ** WE CANNOT SHARE MEMORY with the client
        # ** The client should return the user answer to the questions. 
        # ** We must retrieve this answer in an async way, after some irregular delay
        
        while True:
            try:
                print('Awaiting for result in retrieveResults')
                result = await self.resultQueue.get()
                print(f"Result received")
                           
                for img,name,_ in result:
                    if img is not None:
                        
                        if name == 'unrecognized': 
                            # Ask the user to id the new face 
                            # The answer is sent to retrieveNewFaceId to be processed
                            gui.createGUI_tk(img, self.newFaceIdQueue) 
                            self.guiTime = time.time()  #TODO: in GUI class
                            
                            # TODO: convert the GUI into an object
                        
            except asyncio.CancelledError as e:
                print('The task has been canceled, exit the loop')
                break
            except Exception as e:
                # Handle any other exceptions that might occur
                print(f"Error in run() function: {e}")
                # Notify the queue that the task is done
            finally:
                self.resultQueue.task_done()

    async def processNewFaceId(self): # Receive infos from GUI for id of unrecognized faces
    
        while True:
    
            print('We are awaiting for the new face id.') # for an non-async queue sending data to newFaceIdThread
            newFaceId = await asyncio.get_event_loop().run_in_executor(self.newFaceIdThread,
                                                                    self.newFaceIdQueue.get)
            faceName, faceImg = newFaceId
            
            print(f'We got {faceName}   to process.')
            print(type(faceImg))

            # The image is saved in the directory 'name' (which is created if needed)        
            fl.saveNewFaceImg(faceName, faceImg )
        
            # Update face embeddings for the new face name
            self.sFace.updateFaceEmbedding(faceName, faceImg)
 
            # Re-train the kNN classifier with the updated data
            
            # (?)TODO  For now it is silly: we simply convert all face embeddings again ! 
            # (but I think it is fast enough to justify avoiding the complications... )    
            self.sFace.stackFaceEmbeddingsInArray()  # recompute and set self.X
            self.knn.train(self.X, self.y)
            print(f'We just retrained the kNN layer to (better recognize {faceName}.')

            self.unrecognitionCriteria.init(self.sFace.featuresDict,self.faceNames)
            print('We just recomputed the distance-based criteria for unrecognized faces.')
            
    # ================  Preparation ============================================
    def returnFaceNames(self):
        try: 
            #Default: list face_names i.e. directory names in DATAPATH, except the ones that end with '_new'
            return fl.listFaceNames() # ex: ['audrey', 'francois', 'victor', ']
        except Exception as e:
            print(e)
    # =============================================================================================
    def recognizeFace(self, newFaceImg):
        """ 
        Returns tuple (predicted_faceName, prediction_probability)
        
        We compute it as the faceName associated with the metric whose predictor 
        has the best predicted_proba      
        """
        newFace_features = self.sFace.recognizer.feature(newFaceImg) # array (1, 128)
        predictknn =self.knn.predictKNN
        predict_proba = dict()
        try: 
            #Compute {metric : (faceName, predict_proba) for metric in [l2,cosine]}         
            for metric in ['l2', 'cosine']:
                predicted_index, pred_proba = predictknn(newFace_features, metric )
                predict_proba[metric] = (self.faceNames[predicted_index], pred_proba )

            faceNames_proba_list = list(predict_proba.values()) # [(faceName0, proba0), (faceName0, proba0)]
            print(faceNames_proba_list)
            
            # faceName such that proba is max in [(faceName_l2, proba_l2), (faceName_cos, proba_cos)]
            (faceName, faceName_prob), index = u.argmax_tupls(faceNames_proba_list )
            metric = ['l2', 'cosine'][index] # metric giving the best result above 
            
            print(f'We recognize the face of {faceName} with prob={faceName_prob}')
            
            # False if unrecognized, according to the distance threshold criterion
            recognized =self.unrecognitionCriteria.isRecognized # boolean function
            if not recognized(newFace_features,faceName,metric):
                print('Finally the face is classified as unrecognized.')
                return 'unrecognized', None
            print(f'Cassified as recognized: {faceName} ')   
        except IndexError as ie: 
            print('in recognizerFace', ie)
            faceName = 'unrecognized'
            faceName_prob = 1
        except Exception as e:
            print('in recognizerFace',e)
            faceName = 'unrecognized'
            faceName_prob = 1
              
        finally: 
            return faceName, faceName_prob
             
         
    ## =========  tests   ==========================================
    async def test_runTask(self):
        """To test the asyncio/process/threads structure
        Without doing anything but passing numbers as data"""
        while True:
            faceImg = await asyncio.get_event_loop().run_in_executor(self.executor, 
                                                                    self.inputQueue.get)
            await asyncio.sleep(0.1)  # simulate processing time
        
            result = 'stranger' # faceImg
            print(f"Task is done: result: {result}")
            await self.resultQueue.put(result) 




    
