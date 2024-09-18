# -*- encoding: utf-8 -*-
 
import numpy as np
import cv2 as cv

import os,sys,time

from file_management import YUNET_DETECTION_PATH
import file_management as fl

class FaceDetection():
    algoName = 'yunet' 
    def __init__(self, video=None, score_threshold=0.65, nms_threshold = 0.3, top_k = 5000):
        
        """    score_threshold:  Keeps faces with score >= score_threshold.
            nms_threshold :   Suppress bounding boxes of iou >= nms_threshold.
            top_k = 5000  :   Keep top_k bounding boxes before NMS.
        """    
        self.score_threshold= score_threshold
        self.nms_threshold  = nms_threshold
        self.top_k = top_k
        
        self.modelPath = YUNET_DETECTION_PATH
        
        self.detector =  self._createFaceDetector_yunet()
        
        # withVideo=True when it is a video and not an image that can be of variable size
        if isinstance(video,cv.VideoCapture): 
            self.video = video
            self.setVideoInputSize()
        
        
        self.faces =[]        # list of detected face boxes   

       
    def _createFaceDetector_yunet(self):
        """ 
        .
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

    def setVideoInputSize(self):
        # When we process a video, not an image: 
        # we set the input size right after the detector creation
        scaleFactor = 1.2 
        if isinstance(self.video,cv.VideoCapture):  
            imgWidth = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)* scaleFactor)
            imgHeight = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)* scaleFactor)
            self.detector.setInputSize([imgWidth, imgHeight])
            print('Input size has been adjust to the video.')
        else: 
            print('''It is not a video capture. We cannot adjust the input size yet 
                  since not all images have same size. ''')
                    
                    
    def selectLargestFace(self,faceArray):
        """returns the index of the largest face box (in surface width*height)

        TODO : documentation:  distinguer box & face
        Args:
            faces (_type_): _description_
        """
        if faceArray is not None:
            w = faceArray[:,2]
            h = faceArray[:,3]
            return np.argmax(w*h, 0) 
        return 0    


    def _convertTuple_into_int16(self, tupl):
        """
        Converts of tuple of numbers (e.g. float) into a tuple of 16-bit integers. 
        
        Rem:  for np.ndarray, it is simpler to use .astype(np.int16)     
        
        tupl:  a tuple of number, float or int of more than 16-bits
        
        returns: 
            A tuple of round number casted into 16 bits integers
            
        """
        roundTupl = tuple(round(x) for x in tupl)
        
        # Each value is converted to a 16-bit integer using the bitwise & operator 
        # with a 0xFFFF mask that removes the high bits.
        return tuple(int(x) & 0xFFFF for x in roundTupl)
        

    #TODO : etre certain d'ajuster pour les types (kalman a besoin de np.float32, mais qu' 
    # envoie-t-on au microcontroleur ? 
    # Je pensais envoyer int16, mais si on controle les servos par microseconde, 
    # il est peut-etre mieux de choisir des float32 ?)
    
    def returnBoxCenter(self,box):
        """ Return the center of the box (rectangle enclosing the selected item/face)
        in 16 bits

        Args:
            box : np.ndarray [x,y,w,h] OR tuple (x,y,w,h): selected face to track

        Returns:
            tuple (np.int16, np.int16): Center (x,y) of the box
        """
        x, y, w, h =  box
        if isinstance(box, np.ndarray): 
            return (np.round((x + w//2)).astype(np.int16), 
                    np.round((y + h//2)).astype(np.int16) 
                )
        elif isinstance(box, tuple):
            return self._convertTuple_into_int16( (x + w//2, y + h//2)) 
                
        
    def returnFaceCenter(self,faces, idx, mode = 'eyeCenter'):
        """_summary_

        Args:
            faces : np.array of size (faceNb,15): array of faceNb faces
            idx :int: index of the selected face we want to track

        Returns:
            (np.int16, np.int16): coordinates of the face center in the visual field 
            The face center can either be the eyeCenter OR the rectangle center 
        """
        
        if faces is not None:
        
            if mode == 'eyeCenter':
                x_eye1,y_eye1, x_eye2, y_eye2 = faces[idx,4:-7]#.astype(np.int32)
                x_center = (np.round((x_eye1 + x_eye2 )/2 )).astype(np.int16)  
                y_center = (np.round((y_eye1 + y_eye2 )/2 )).astype(np.int16)  
                return (x_center, y_center) 

            elif mode == 'rectCenter': 
                return self.returnBoxCenter(faces[idx,:4])
        
        return None
            

        
    # TODO ***** ??    
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