# -*- encoding: utf-8 -*-


import time

import numpy as np
from numpy.linalg import norm 

from filtering_oo import Filtering


class Trajectory:
   
    def __init__(self, modeLabel, initPt=None):
        self.format = np.int16 # To be changed eventually according to the servo-control code in the microcontroller
        self.modeLabel = modeLabel
        
        # Kalman Filtering is used to smooth the observed trajectory of face centers
        self.filter = Filtering(modeLabel, initPt) # Class for the Kalman Filter
        
        self.observations = list()   # faceCenterTraject   # = Filtering.measurements
        self.smoothObs = list()                          # = Filtering.prediction 
        self.observationsTime = list() 

                  
    def getMode(self):
        return self.modeLabel
        
    def setFormat(self,format):
        self.format = format
            
    def reinit(self):
        """ 
        Reinitialize the Kalman filter state, the observations and the filtered observations 
        It does not create a new Kalman filter.
        Arg: 
            lastPrediction : np.array() :  x, y, vx, vy; 
                    i.e last predicted state and its predicted velocity 
                    If None, the class variable is used instead. 
        """
        self.observations = list()   
        self.smoothObs = list()    
        self.observationsTime = list() 
    
        try:    
            self.filter.setKalmanInitialState() # lastPrediction =x, y, vx, vy 
        except Exception as e:
            print(e)

      
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
        self.smoothObs.append(self.filter.predictions[-1].astype(self.format))
        
    
    def isAtFirstStep(self):
        return len(self.observations)==1
     
    
    def inFastMotion(self):   # TODO
        return False     

    def getLastSmoothPt(self):
        """  Return  x, y from the last filtered observation (x,y,vx,vy)"""
        # The difference is only  np.int16  vs  np.float32
        #print(self.smoothObs[-1][:2] )
        #print(Filtering.lastPrediction )
        return self.smoothObs[-1][:2]
    
    def distance(self):
        """ Return the l2 distance between the current point and the previous one."""
        if len(self.observations) > 2:
            diff = np.array(self.observations[-1]) - np.array(self.observations[-2]) # array 
            return norm(diff.astype(np.float32),ord=2)
        else: 
            return 0
    
    def velocity(self):
        if len(self.observations) > 2:    
            velocityArray = self.smoothObs[-1][2:4]
            return norm(velocityArray.astype(np.float32),ord=2)
        else: 
            return 0
                
    def acceleration(self):
        if len(self.observations) > 2:   
            #print(self.filteredObs[-1][2:4]) 
            diff = np.array(self.smoothObs[-1][2:4]) - np.array(self.smoothObs[-2][2:4]) 
            return norm(diff.astype(np.float32),ord=2)
        else: 
            return 0            
        
        
    def needAcquisition(self):
        """ We need trajectory data to calibrate the Kalman Filter (noise parameters etc)"""
        return False
    
    def acquisition(self, startTime):
        # **** TODO: PENSER A CE QUE L'ON VOUDRAIT FAIRE ICI ***
        acquisitionTime = 60 #[s]   1 min        
        if time.time() - startTime > acquisitionTime:
            # Signal acquisition for state model estimation
    
            # TODO pour a la fois detection que pour tracking
            #fl.saveTraject(detectionTraject, mode)         
            self.reinit()       
        