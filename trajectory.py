# -*- encoding: utf-8 -*-


import time

import numpy as np

import filtering as filt     # Custom module for filtering of face center trajectory (Kalman filter)

class Trajectory:

    def __init__(self):
        self.reinit()
        filter = Filtering
        
                            

        
        
    def reinit(self):
        self.observations = list()   # faceCenterTraject   # = Filtering.measurements
        self.filteredObs = list()                          # = Filtering.prediction 
        self.observedTrajectTime = list() 

        # To init trackingTraject: 
        x0, y0 = self.filter.lastPredCoord[:2] #last filter prediction ( not matter from what mode e.g: from detection)
        trackingFilter,filteredTraject=filt.initFiltering((x0, y0),
                                                    'tracking')

      
    def append(self, observedCenter):
        self.observations.append(observedCenter)
        self.observedTrajectTime.append(time.time())

    def convertInFloat32(observedCenter_int16):
        # TODO: rester consistent dans le choix de type (float32 vs int16 etc)
        # int16 has to be transformed into float32 for the kalman filter to work on it        
        return np.array(observedCenter_int16, np.float32).reshape(2,1) 
        
    def update(self, observedCenter):
        
         # int16 has to be transformed into float32 for the kalman filter to work on it   
        observedCenter = self.convertInFloat32(observedCenter)    
        self.append(observedCenter)            
        
        #Kalman Filtering is used to smooth the observed trajectory of face centers
        filteredTraject = filt.updateFiltering(observedCenter,trackingFilter, 
                                                filteredTraject)
        smoothCenter = filteredTraject[-1] 

        
    def inFastMotion(self):
        return False     