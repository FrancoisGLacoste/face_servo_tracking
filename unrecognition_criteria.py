# -*- encoding: utf-8 -*-



"""  unrecognition_criteria.py

 Distance-based Threshold criteria for the discrimination of the unrecognized category
For distance-based discrimination of the unrecognized category  
"""

import numpy as np 
import cv2 as cv

from sface_embeddings import SFaceEmbeddings

class UnrecognitionCriteria: 
    
    def __init__(self, sFace: SFaceEmbeddings, faceNames: list):
        faceEmbeddings = sFace.featuresDict  # dict
        
        # 
        self.dist_l2 = sFace.dist_l2           # function 
        self.dist_cosine = sFace.dist_cosine   # function 

        self.init(faceEmbeddings,faceNames)

      
    def init(self,faceEmbeddings,faceNames):
        """Prepare the unrecognition criteria 
        Used in self.__init__(), but also in FaceRecognition.processNewFaceId()
        """
        self.centroidsDict   = {name: None for name in faceNames}  
        self.discreteDistribDict = {'l2': dict(), 'cosine':dict()} 
        self.radiusDict      = {'l2': dict(), 'cosine':dict()}
  
        # Criteria is based on a distance threshold and centroids for all faceName      
        self.computeCentroids(faceEmbeddings)   # Centroids of the data clusters 
        self.computeDistDistrib(faceEmbeddings) #   
        self.computeRadius(faceEmbeddings)      # Radius of the data clusters   
    
    
                        
    def computeCentroids(self, faceEmbeddingsDict):
        for faceName in faceEmbeddingsDict.keys():
            xArr =faceEmbeddingsDict[faceName] #  array (Nn, 128)            
            Nn = xArr.shape[0]
            dim =  xArr.shape[1] # =128
            centroid = (np.sum(xArr,0)/Nn).reshape(1, dim)
            self.centroidsDict[faceName] = centroid.astype(dtype=np.float32)
                
       
    def computeDistToCentroid(self, X, faceName, dist ='l2'):
        """Compute the distance dist of X to the centroid of faceName   """
        distFct = {'l2':self.dist_l2 ,'cosine': self.dist_cosine }
        centroid = self.centroidsDict[faceName] 
        
        return distFct[dist](centroid, X)
 
               
    def computeDistDistrib(self,faceEmbeddingsDict):
        """ 
        For each faceName and each distance metric (l2 or cosine):
        Compute the discrete distribution of the distances to the centroid. 
        and the radius (=maximum distance between the centroid and a data point)
        i.e.: 
        It sets:  
            self.distDistribDict[dist][name] : list of increasing distance values
            self.radiusDict[dist][name]      : maximum distance
        """                  
        for dist, distFct in [('l2',self.dist_l2), ('cosine',self.dist_cosine)]:
            for name in faceEmbeddingsDict.keys():
                centroid =  self.centroidsDict[name]
                X_n = faceEmbeddingsDict[name]
                #print(type(centroid))
                #print(centroid.shape)
                distrib = sorted([distFct(centroid,x) for x in X_n])
                self.discreteDistribDict[dist][name] = distrib
                
    def computeRadius(self,faceEmbeddingsDict):
        for dist in ['l2','cosine']:
            for name in faceEmbeddingsDict.keys():
                distrib =self.discreteDistribDict[dist][name]
                self.radiusDict[dist][name] = distrib[-1]            
                        
    
     
                
    def isRecognized(self,X,faceName,dist='l2'):
        """
        Return: 
            True when 'recognized' and we keep the {faceName} prediction of the classifier (kNN...)
            i.e.  if the distance of X to centroid < the maximal distance to centroid (i.e. the radius)
         
            False when 'unrecognized'          
         """        
        if self.computeDistToCentroid(X, faceName, dist) < self.radiusDict[dist][faceName]:
            return True
        else: 
            return False
        
         