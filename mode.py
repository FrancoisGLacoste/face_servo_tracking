# -*- encoding: utf-8 -*-


import time 



class Mode:
    # Constant parameters: 
    minTimeInDetectionMode = 10 # [s]  in UTC time 

    def __init__(self,modeLabel):
        if 'detection' in modeLabel.lower():
            self.modeLabel = 'faceDetection'  # initialize the mode to face detection
        elif 'tracking' in modeLabel.lower():
            self.modeLabel = 'faceTracking'   # initialize the mode to face tracking
        self.startTime = time.time()     # for current UTC time in sec.
 
    def isTimeToSwitchToTracking(self, faces):
        """determine when the mode switches from faceDetection mode to faceTracking mode
        """
        if faces is None: 
            return False
        
        # Stay in detection mode for at least {minTimeInDetectionMode} [ms]
        if self.getModeTime() < self.minTimeInDetectionMode:
            return False
        
        else:
            print('Switch from detection mode to tracking mode now !')
            self.modeLabel = 'faceTracking'
            self.startTime = time.time() # to measure the time in faceTracking mode
            return True 
  
    def switchBackToDetection(self):
        # Switch back to face detection mode
        print(f'''Target is lost: go back to face detection after 
                spending {self.getModeTime()/1000} s in face tracking mode.''')
        self.modeLabel ='faceDetection'
        self.startTime = time.time() # to measure the time in faceDetection mode
        
     
    def reinitTime(self):
        self.startTime = time.time()
            
    def get(self):
        return self.modeLabel  
    
    def getModeTime(self):
        print('Time elapsed in the current mode:', time.time() - self.startTime)
        return time.time() - self.startTime  
    
    def isInDetectionMode(self):
        """  Return Bool"""
        return self.get() == 'faceDetection'
    
    def isInTrackingMode(self):
        return self.get() == 'faceTracking'
    