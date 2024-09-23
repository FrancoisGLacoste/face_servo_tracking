# -*- encoding: utf-8 -*-


import time, sys





class Mode:
    # Constant parameters: 
    minTimeInDetectionMode = 5 # sec

    def __init__(self,modeLabel):
        self._modeTime = 0                    # initialize mode time to 0
        if 'detection' in modeLabel.lower():
            self.modeLabel = 'faceDetection'  # initialize the mode to face detection
        elif 'tracking' in modeLabel.lower():
            self.modeLabel = 'faceTracking'   # initialize the mode to face tracking
        self.startTime = time.time()
 
    def isTimeToSwitchToTracking(self, faces):
        """determine when the mode switches from faceDetection mode to faceTracking mode
        """
        if faces is None: 
            return False
        
        # Stay in detection mode for at least {minTimeInDetectionMode} sec
        if self._modeTime < self.minTimeInDetectionMode:
            return False
        
        else:
            self.modeLabel = 'faceTracking'
            print('Switch from detection mode to tracking mode now !')
            self.startTime = time.time() # to measure the time in faceTracking mode
            return True 
  
    def switchBackToDetection(self):
        # Switch back to face detection mode
        print(f'''Target is lost: go back to face detection after 
                spending {self.getModeTime()} s in face tracking mode.''')
        self.modeLabel ='faceDetection'
        self.startTime = time.time() # to measure the time in faceDetection mode
        
    def get(self):
        return self.modeLabel  
    
    def getModeTime(self):
        return self.startTime - time.time()  
    
    def isInDetectionMode(self):
        """  Return Bool"""
        return self.get() == 'faceDetection'
    
    def isInTrackingMode(self):
        return self.get() == 'faceTracking'
    