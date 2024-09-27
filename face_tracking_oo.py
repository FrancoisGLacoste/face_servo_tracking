# -*- encoding: utf-8 -*

"""
face_tracking.py
"""
import os 

import numpy as np
import cv2 as cv

from file_management import VIT_TRACKING_PATH 

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
        self.tm = cv.TickMeter() 
        self.tracker = self._createFaceTracker_vittrack()
        self.isSuccessful = False
        self.score = None
        
    def _createFaceTracker_vittrack(self):   
        
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
        face: np.array(4,) of np.float32: Coord [x,y,w,h] of the face to track
        """ 
        #change the type of faces:  np.ndarray of shape (4,) --> np.int16
        face_int16 = np.round(face).astype(np.int16)              
        self.tracker.init(img,tuple(face_int16))     
        


    def track(self, img: np.ndarray):
        """  img: cv2.typing.MatLike, Umap"""
        self.tm.reset() # to monitor tracking algo performance 
        self.tm.start()
        self.isSuccessful, faces = self.tracker.update(img) # tuple[bool, cv2.typing.Rect]
        self.score = self.tracker.getTrackingScore()
        self.tm.stop()
        return faces   # faceTuple
