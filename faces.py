# -*- encoding: utf-8 -*-

import cv2 as cv
import numpy as np

class Face:
    
    def __init__(self, frameId, box, score, tm: cv.TickMeter, mode='detection' ):
        self.frameId =frameId # = timeStep in the faceDetection/faceTracking loop
        self.box = box  # (x,y,w,h)
        self.fps =tm.getFPS()   # apriori not the same as cv.CAP_PROP_FPS
        self.time = tm.getTimeMilli()   # Either detection time or tracking time [ms]
        #avgDetectTime = tm.getAvgTimeMilli()  # when tm is not reset 

        self.mode = mode  # either detection OR tracking 
        self.score = score # Either detection score OR tracking score
        self.observedCenter = (None,None)
        
        if box is not None: 
            self.observedCenter = self.returnBoxCenter(box)
        
        # Only for the active face , which is involved in the filtered trajectory
        self.smoothCenter = (None,None)
    
        # Only the faces in detection mode, which are "piped" to face recognition. 
        # (recognizedName, proba) # TODO BUT DOES IT MAKE SENSE . IS IT USEFUL ?? HOW TO DO IT??
    
    
    def returnBoxCenter(self,box):
        """ Return the center of the box (rectangle enclosing the selected item/face)
        in 16 bits

        Args:
            box : np.ndarray [x,y,w,h] OR tuple (x,y,w,h): selected face to track

        Returns:
            tuple (np.int16, np.int16): Center (x,y) of the box
        """
        
        x, y, w, h =  box
        if isinstance(box, np.ndarray): 
            return (np.round((x + w//2)).astype(np.int16), 
                    np.round((y + h//2)).astype(np.int16) 
                )
        elif isinstance(box, tuple):
            return self._convertTuple_into_int16( (x + w//2, y + h//2)) 
                
    
    
    def _convertTuple_into_int16(self, tupl):
        """
        Converts of tuple of numbers (e.g. float) into a tuple of 16-bit integers. 
        
        Rem:  for np.ndarray, it is simpler to use .astype(np.int16)     
        
        tupl:  a tuple of number, float or int of more than 16-bits
        
        returns: 
            A tuple of round number casted into 16 bits integers
            
        """
        roundTupl = tuple(round(x) for x in tupl)
        
        # Each value is converted to a 16-bit integer using the bitwise & operator 
        # with a 0xFFFF mask that removes the high bits.
        return tuple(int(x) & 0xFFFF for x in roundTupl)
        

        #TODO : etre certain d'ajuster pour les types (kalman a besoin de np.float32, mais qu' 
        # envoie-t-on au microcontroleur ? 
        # Je pensais envoyer int16, mais si on controle les servos par microseconde, 
        # il est peut-etre mieux de choisir des float32 et de discretiser seulement dans le controleur?)
    


    def dict(self):
        """  We transform the content of Face object into a dictionary
        to be able to serialized it when we send it through a queue or a pipe, 
        or through sockets.
        
        __dict__ is a dictionary containing all attributes of a given object.
        """
        return self.__dict__ # a dictionary, not a pickle yet
        
    # Remark : we should be able to rebuild the objects using the constructor
    def rebuild(self,dict):
        return self.__init__(*dict.get())    