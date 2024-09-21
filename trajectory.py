# -*- encoding: utf-8 -*-


import time

import filtering as filt     # Custom module for filtering of face center trajectory (Kalman filter)

class Trajectory:

    def __init__(self):
        self.reinit()
        filter = 
        
    def reinit(self):
        self.observations = list()   # faceCenterTraject
        self.filteredObs = list()
        self.observedTrajectTime = list() 
      
    def append(self, observedCenter):
        self.observations.append(observedCenter)
        self.observedTrajectTime.append(time.time())
    
    def inFastMotion(self):
        return False     