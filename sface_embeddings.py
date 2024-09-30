# -*- encoding: utf-8 -*-

""" 
sface_embeddings.py

computation of the SFace features Embeddings
"""        
import os 
 
import cv2 as cv
import numpy as np

from file_management import BASE_DIR
import file_management as fl

class SFaceEmbeddings:
    
    def __init__(self, faceNames):
        self.faceNames = faceNames
        self.y = np.array(range(len(self.faceNames))) # class labels are indices in faceNames

        self.recognizer = cv.FaceRecognizerSF.create(
                            os.path.join(BASE_DIR,'face_recognition_sface_2021dec.onnx'), 
                            "" )
        
        self.featuresDict ={name: list() for name in faceNames} 

        self.prepareSFaceEmbeddings()
        
        
    def prepareSFaceEmbeddings(self): 
        
        # Compute or load self.featuresDict[face_name] 
        if not self.embeddingsAreSaved():
            self.computeAllFaceEmbeddings() 
        else: 
            self.loadEmbeddings()
    
    
    
    # =========================    Face feature embeddings ==============================    
    def computeFaceEmbeddings(self, face_names):
        """Train SFace algorithm by computing face embeddings.
        
        It prepares a dictionary of the embeddings of face features of each face name
        self.featuresDict = { face_name1:  array(N0, dim), 
                                    face_name2:  array(N1, dim),
                                    ...}
                                    where   Nn = number of images 
                                            dim = embedding dimension (=128)
        """
        for face_name in face_names:
            embeddings = []
            for face_img in fl.yieldFaceImgs(face_name):     #np.ndarray
                # REM: ils utilisaient aligncrop() pour obtenir face_img
                embedding = self.recognizer.feature(face_img)
                #print(embedding.shape)   # (1,128)
                dim = embedding.shape[1]
                embeddings.append(embedding)
            
            Nn = len(embeddings)   
            self.featuresDict[face_name]  = (np.array(embeddings)).reshape(Nn, dim)
            
        
    def computeAllFaceEmbeddings(self):
        """ 'Train' SFace algorithm by computing the face embeddings 
           for all face names that exists in DATAPATH   
        """
        try:
            self.computeFaceEmbeddings(self.faceNames)
            print('We have just computed all face embeddings.')
        except Exception as e: 
            print(e)    
                
    def updateFaceEmbedding(self, faceName, faceImg): 
        """  Updates face embeddings with the last new face image that have just been 
        identified by the user. 
        
        """
        # Compute the feature embedding of the new face image
        X = self.recognizer.feature(faceImg)   # (1,128)  
    
        # In case we never met 'faceName' before:
        if faceName not in self.featuresDict.keys(): 
            self.featuresDict[faceName] = X
        else:         
            # stack them
            X_n = self.featuresDict[faceName]  # (Nn, 128)
            self.featuresDict[faceName] = np.vstack([X,X_n]) # (Nn+1, 128)
        print('We just add the face embeddings to the new {faceName} image.')
            
   
    def stackFaceEmbeddingsInArray(self):
        """   
        Face Embeddings is a dictionary of array of image feature embeddings: 
                            {'audrey': array(N0,128), 'francois':array(N1,128)}
                            where img_embed1, img_embed2, ...are np.array(1,d) , d=128 dimensions
            
        But sci-kit-learn methods expect the dataset to be in np.arrays
        i.e. 
        X : np.array(Nn, dim) : data points (i.e.faceEmbeddings) for a given label (i.e. face name)
        y : np.array(Nn,)     : array of label indices ( i.e. each index represents a face name)
        """
        
        for n, faceName in enumerate(self.featuresDict.keys()):
            #embed_list = self.featuresDict[faceName] # list of array(1,128)
            #y_n = n*len(embed_list)
            X_n = self.featuresDict[faceName]
            Nn = X_n.shape[0]
            y_n = (np.array([n]*Nn)).reshape(Nn,)      # array(Nn, )
            
            # Stack the arrays X <--- [ X|X_n ]
            if n == 0:
                X = X_n
                y = y_n 
            else:
                X = np.vstack([X,X_n])
                y = np.hstack([y,y_n])
                
        #return X,y
        self.X = X
        self.y = y 
 
    # TODO ?        
    def embeddingsAreSaved(self):
        """ 
        Returns a Boolean value: True when we find a file that contains the SFace features embeddings 
        
        """
        # TODO
        return False
    
    def loadEmbeddings(self):
        # TODO
        NotImplemented
        
    
    #=== used in the unrecognition_criteria ====================
    
    def dist_l2(self,features1, features2=None):  
        features1 = features1.reshape(1,128).astype(dtype=np.float32)
        if features2 is None: 
            features2 = np.zeros(features1.shape)
        features2 = features2.reshape(1,128).astype(dtype=np.float32) 
        dist1 = self.recognizer.match(features1, features2,cv.FaceRecognizerSF_FR_NORM_L2)
        #dist2 = np.linalg.norm(features1 - features2, ord=2).astype(dtype=np.float32)  # OK dist2 == dist1
        #dist3 = np.sum((features1 - features2)**2).astype(dtype=np.float32)   # square of l2 dist
        return dist1   # dist2 and np.square(dist3)  are equivalent 
    
    def dist_cosine(self,features1, features2=None):
        features1 = features1.reshape(1,128).astype(dtype=np.float32)
        if features2 is None: 
            features2 = np.zeros(features1.shape)
        features2 = features2.reshape(1,128).astype(dtype=np.float32)
        return self.recognizer.match(features1, features2,cv.FaceRecognizerSF_FR_COSINE)     
        
     