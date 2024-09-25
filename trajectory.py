# -*- encoding: utf-8 -*-


import time

import numpy as np
from numpy.linalg import norm 

from filtering_oo import Filtering
#import filtering_oo as filt    

class Trajectory:

    # Class variable that is shared between the detection trajectory and the tracking trajectory
    lastSmoothPt = None
    
    def __init__(self, modeLabel, initPt=None):
        self.format = np.int16 # To be changed according to servo-control code in the microcontroller
        self.modeLabel = modeLabel
        
        # Kalman Filtering is used to smooth the observed trajectory of face centers
        self.filter = Filtering(modeLabel, initPt) # Class for the Kalman Filter
        
        self.observations = list()   # faceCenterTraject   # = Filtering.measurements
        self.filteredObs = list()                          # = Filtering.prediction 
        self.observationsTime = list() 

                  
    def getMode(self):
        return self.modeLabel
        
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
        self.observations = list()   
        self.filteredObs = list()    
        self.observationsTime = list() 

        if lastPrediction is None : 
            lastPrediction = self.lastSmoothPt 
        
        if lastPrediction is not None:     
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
        #print(type(self.observations))
        #print(lastObservation)
        #print(type(lastObservation))
        self.filter.update(lastObservation)
        self.filteredObs.append(self.filter.predictions[-1].astype(self.format))
        
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
    
    def distance(self):
        """ Return the l2 distance between the current point and the previous one."""
        if len(self.observations) > 2:
            diff = np.array(self.observations[-1]) - np.array(self.observations[-2]) # array 
            return norm(diff.astype(np.float32),ord=2)
        else: 
            return 0
    
    def velocity(self):
        if len(self.observations) > 2:    
            velocityArray = self.filteredObs[-1][2:4]
            return norm(velocityArray.astype(np.float32),ord=2)
        else: 
            return 0
                
    def acceleration(self):
        if len(self.observations) > 2:   
            #print(self.filteredObs[-1][2:4]) 
            diff = np.array(self.filteredObs[-1][2:4]) - np.array(self.filteredObs[-2][2:4]) 
            return norm(diff.astype(np.float32),ord=2)
        else: 
            return 0            