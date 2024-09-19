# -*- encoding: utf-8 -*

"""
visual_tracking.py
"""
import os 

import numpy as np
import cv2 as cv

from file_management import VIT_TRACKING_PATH 

#fl.BASE_DIR = '/home/moi1/Documents/dev_py/vision/PROJET_Face-Tracking-camera/'

def isCuda():  
    try:
        return (cv.cuda.getCudaEnabledDeviceCount() > 0)
       
    except:
        return False


#=====================================================================
#       Tracking with vittrack
#    https://docs.opencv.org/4.x/d9/d26/classcv_1_1TrackerVit.html
#    https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
#=====================================================================

class FaceTracking():
    def __init__(self):
        self.tracker = self.createFaceTracker_vittrack()
        
    def createFaceTracker_vittrack(self):   
        
        if isCuda():
            backend = 1 # i.e. cv.dnn.DNN_BACKEND_CUDA
            target  = 1 # i.e. cv.dnn.DNN_TARGET_CUDA

        else :
            backend = 0 # i.e. cv.dnn.DNN_BACKEND_OPENCV
            target  = 0 # i.e. cv.dnn.DNN_TARGET_CPU
        
        param = cv.TrackerVit.Params()
        param.net     = VIT_TRACKING_PATH
        param.backend = backend
        param.target  = target 
        tracker = cv.TrackerVit.create(param)
        return tracker
    
    
    def initTracker(self, img, face):
        """
        face:     np.ndarray of shape (4,)  :  face to track
        """ 
        self.tracker.init(img,tuple(face))     
        


    

