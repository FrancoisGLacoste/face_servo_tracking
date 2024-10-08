# -*- encoding: utf-8 -*-

import multiprocessing as mp
 
import numpy as np
import cv2 as cv

from file_management import YUNET_DETECTION_PATH
import file_management as fl
from faces import Face
from trajectory import Trajectory
from face_recognition_SFace_oo import FaceRecognition
 
class FaceDetection():
    algoName = 'yunet' 
    def __init__(self, video=None, score_threshold=0.65, nms_threshold = 0.3, top_k = 5000):
        
        """    score_threshold:  Keeps faces with score >= score_threshold.
            nms_threshold :   Suppress bounding boxes of iou >= nms_threshold.
            top_k = 5000  :   Keep top_k bounding boxes before NMS.
        """    
        
        # Parameters of the Yunet detector
        self.tm = cv.TickMeter() # To measure computation times
        self.step = 0
        self.score_threshold= score_threshold
        self.nms_threshold  = nms_threshold
        self.top_k = top_k
        self.modelPath = YUNET_DETECTION_PATH
        
        self.detector =  self._createFaceDetector_yunet()
        self.isSuccessful = False
        
        # Set dimensions and size ( bytes) of the frames to detect
        if isinstance(video,cv.VideoCapture): 
            (self.imgWidth, self.imgHeight), self.frameSize = self.returnFrameSize(video)
            self.detector.setInputSize([int(self.imgWidth), int(self.imgHeight)])
        else:
            self.imgWidth, self.imgHeight, self.frameSize   = None,None,None
            print(f'No video , width, height and size are set to None.')
                     
  

    def _createFaceDetector_yunet(self):
        """ 
         Factory that returns Yunet detector objects
        """
        faceDetector = cv.FaceDetectorYN.create(
            self.modelPath,
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k
        )
        return faceDetector

    def returnFrameSize(self, video: cv.VideoCapture):
        scaleFactor =1 
        try:  
            imgWidth = video.get(cv.CAP_PROP_FRAME_WIDTH)* scaleFactor
            imgHeight = video.get(cv.CAP_PROP_FRAME_HEIGHT)* scaleFactor
            channels = 3 # assuming GBR
            frameSize = int(imgWidth) * int(imgHeight) * channels
            
            return (imgWidth, imgHeight), frameSize
        except Exception as e: 
            print(f'Error: {e}')
                      
    # ===================================================================================
    def detect(self, img):
        """ 
        Returns: list of Face objects (not necessarily in order of size )
        """            
        self.tm.reset()    
        self.tm.start()  # TODO :  can we have processingTime as member of FaceDetector ?
        self.isSuccessful, facesArray = self.detector.detect(img) 
        self.tm.stop()
        
        faces = self.createFaceObjectList(self, facesArray)   
        activeFaceIndex = self.selectLargestFaceIndex(self, facesArray)
        return faces, activeFaceIndex
    
    
    # ===================================================================================
    
    def createFaceObjectList(self, facesArray):    
        faces = list()
        frameId = self.step
        for faceArray in facesArray:
            score = faceArray[-1]
            coords = faceArray[:-1]   # transform in .astype(np.int32)  only when visualization              
            x,y,w,h,x_eye1,y_eye1,x_eye2,y_eye2,x_nose,y_nose,x_mouth1,y_mouth1,x_mouth2,y_mouth2 = coords
            faces.append(Face(frameId, (x,y,w,h), score, self.tm))
        return faces
    
            
    def selectLargestFaceIndex(self, faceArray):
        """returns the index of the largest face box (in surface width*height)

        TODO : documentation:  distinguer box & face
        Args:
            faces (Array): _description_
        """
        if faceArray is not None:
            w = faceArray[:,2]
            h = faceArray[:,3]
            return np.argmax(w*h, 0) 
        return 0    

 
                    
    # =======================================================================================
                
    def recognitionCondition(self,detectionTraject: Trajectory):   
        
        self.step +=1
              
        #------ ??? DEVRAIS-JE CREER UNE NOUVELLE CLASSE 'Face' (ou kekchose du genre) 
        # On veut comparer le centre de la face active courante  avec la precedente face: centre, select_idx
        # Parfois j'ai [francois, francois, unrecognized, francois,...]
        # Ou pire: [francois, francois, audrey, francois,...]
        # On veut se baser sur la continuite des centres des images pour conclure 
        # a la continuite des faceName 
        # (c-a-d que ci-haut, audrey et unrecognized devraient etre francois)
        # Mais ca s'infere seulement a posteriori. On peut pas deviner sur le champs !
        nearPreviousFace  = (detectionTraject.distance() < 6)  
        likelyTheSameFace = bool(  np.mod(self.step+1,5 )) and nearPreviousFace  

        # Le probleme sera aussi de detecter un saut d'une face a une autre quand il y 
        # en a plus d'une.... 
        # Et on voudra ne permettre le saut qu'a certaines conditions. 
        
        condition =  (self.step ==1 or not likelyTheSameFace )   \
                        and not detectionTraject.inFastMotion()        
        # Rem: This condition does not take into account constraints on faceRecognition properties 
        return condition
    
    '''    
    def sendToFaceRecognition_v2(self, img, faces ,
                              faceRecognition :FaceRecognition):
        """ 
        img  :    UMap, cv2.typing.MatLike or np.array : the whole camera frame (image)
        faces:    np.array [:,:4] :  a sequence of boxes, one box for each face    """
        #Rem: detection output: faces is array[(faceNb,15)]: array of faceNb faces
        # TODO:  devrais mettre cette fct dans une classe ??
        face_imgs = faceRecognition.cropBoxes(img, faces)   
        """self.recognizer.alignCrop(src_img: cv2.typing.MatLike, face_box: cv2.typing.MatLike, aligned_img: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
            
            recognizer.alignCrop(src_img: UMat,               face_box: UMat,               aligned_img: UMat               | None = ...) -> UMat: ...
        """
        print('Transferring image to face recognition module via shared memory.')
    '''   
    '''
    # TODO ***** ??=======================================================================    
    def increaseBox(self,box):  
        # Because when detecting in image we must adjust the scaling for each image
        x, y, w, h = box 
        #new_x
        NotImplemented
        
    def increaseBoxes(self,box_list):
        new_box_list =list()
        for box in box_list:
            new_box_list.append(self.increaseBox(box))
        return new_box_list
    '''    
        
    # =======================================================================================    
    def detectFacesFromFile(self,face_name ='unknowns', number=-1):
        """  Each photo apriori has different size, which is not the video's one,"""
        # List of new images with non-extracted faces from directory face_name+'_new'
     
        
        new_imgs = fl.readImgFiles(face_name+'_new', number)
        cnt =0
        face_names=list()
        print('Begining face detection')
        for (id, img) in new_imgs:
            # Detect and crop the faces 
            #imgHeight, imgWidth = img.shape[:2]
            #print(type(img.shape[:2]))
            
            #print(list(reversed(img.shape[:2])))
            self.faceDetector.setInputSize(list(reversed(img.shape[:2])))
            #faceDetector.setInputSize([imgWidth, imgHeight])
            isSuccesful, face_boxes = self.faceDetector.detect(img) 
            
            # On voudrait agrandir les boites pour capturer tout le visage...
            #face_boxes = increaseBoxes(face_boxes)
            
            if not isSuccesful or face_boxes is None: 
                continue
            if len(face_boxes)==0:
                continue
            face_imgs = fl.cropBoxes(img, face_boxes)     
            imgNb = len([im for im in face_imgs if im is not None])
            
            # We consider that image id has 'number' face images of name face_name 
            face_names = [face_name ]*imgNb
            file_id = [id]*imgNb
            if imgNb == 0 :
                print(f'No face detected in this image... nothing to save ! ')
                continue    
            notSaved = fl.save_face_imgs(face_names,face_imgs, file_id)
            cnt+=imgNb
        print(f'Detection is complete: saved a total of {cnt} faces in {len(new_imgs)} images.')        
   
    def test_detectFromFile(self):
        # Detect and crop all faces from image files in 'audrey_new' directory, 
        # and save the cropped images in 'audrey' directory
        self.detectFacesFromFile('audrey')   
    
    
if __name__ == '__main__':
    faceDetection = FaceDetection()
    faceDetection.test_detectFromFile()