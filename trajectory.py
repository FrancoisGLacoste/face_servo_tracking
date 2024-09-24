# -*- encoding: utf-8 -*-


import time

import numpy as np

from filtering_oo import Filtering
#import filtering_oo as filt    

class Trajectory:

    # Class variable that is shared between the detection trajectory and the tracking trajectory
    lastSmoothPt = None
    
    def __init__(self, modeLabel, initPt=None):
        self.format = np.int16 # To be changed according to servo-control code in the microcontroller
        
        # Kalman Filtering is used to smooth the observed trajectory of face centers
        self.filter = Filtering(modeLabel, initPt) # Class for the Kalman Filter
        
        self.reinit(initPt)                    

        
    def setFormat(self,format):
        self.format = format
            
    def reinit(self, lastPrediction = None):
        """ 
        Reinitialize the Kalman filter state, the observations and the filtered observations 
        It does not create a new Kalman filter.
        Arg: 
            lastPrediction : np.array() :  x, y, vx, vy; 
                    i.e last predicted state and its predicted velocity 
                    If None, the class variable is used instead. 
        """
        self.observations = list()   # faceCenterTraject   # = Filtering.measurements
        self.filteredObs = list()                          # = Filtering.prediction 
        self.observationsTime = list() 

        if lastPrediction is None : 
            lastPrediction = self.lastSmoothPt 
        self.filter.setKalmanInitialState(*lastPrediction) # lastPrediction =x, y, vx, vy 
        

      
    def appendObs(self, newObservation):
        self.observations.append(newObservation)
        self.observationsTime.append(time.time())
        
    '''
    def convertInFloat32(observedCenter_int16):
        # TODO: rester consistent dans le choix de type (float32 vs int16 etc)
        # int16 has to be transformed into float32 for the kalman filter to work on it        
        return np.array(observedCenter_int16, np.float32).reshape(2,1) 
    '''    
    
    def updateFilter(self):
              
        lastObservation = self.observations[-1]
        self.filter.update(lastObservation)
        self.filteredObs = self.filter.predictions.astype(self.trajectFormat)
        
        self.setLastPrediction()
        
    
    def isAtFirstStep(self):
        return len(self.observations)==1
    
    def setLastPrediction(self):
        # To share the last prediction across the trajectory instances
        # i.e. from detection trajectory to tracking trajectory and vice-versa
        self.lastSmoothPt = self.filter.predictions[-1]
        
    def inFastMotion(self):   # TODO
        return False     

    def getLastSmoothPt(self):
        """  Return  x, y from the last filtered observation (x,y,vx,vy)"""
        return self.filteredObs[-1][:2]