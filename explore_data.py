# -*- encoding: utf-8 -*-

"""
explore_data.py 


"""
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.decomposition import PCA 

from face_recognition_SFace_oo import FaceRecognition

# ====================    Density Estimation =========================================
# Problem is some classes have too few data points to allow a meaningful estimation

def estimateDistDensity(faceRecognition: FaceRecognition):    # TODO: smoothing hyperparameter ?
    # Not Necessarily useful but can be interesting to understand the data
    """ 
    Estimate the density distribution of distances to centroid 
    with kernel density (unsupervised learning)
    Only makes sense when Np large enough (Np > Np_thresh )
    
    discreteDistrib : list  (like self.distDistribDict[dist][name] )
    """
    
    Np_thresh = 8
    for dist in ['l2','cosine']:
        for name in faceRecognition.faceEmbeddingsDict.keys():
            distancesList = faceRecognition.discreteDistribDict[dist][name]
            
            Np = len(distancesList)
            if Np >= Np_thresh:
                X = np.array(distancesList)  # array of distances
                
                # TODO Choice of the smoothing parameter: 
            
                
                h=0.2   
                density = KernelDensity(kernel ='gaussian', bandwidth=h).fit(X)
                KernelDensity[dist][name] =density 




## ============================== For (face features) data visualization ===========================         
    
def project3D_PCA(self):
    
    X = self.convert
    pca = PCA(n_component =3,whiten=False )
    data = pca.fit_transform(X)
    # explained_variance_ratio_