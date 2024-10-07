
# -*- encoding: utf-8 -*-

"""  
image_display_v3.py
"""
from trajectory import Trajectory
from result_transfer import ResultTransfer

import visualization as v
from visualization import GREEN, BLUE


class ImageDisplay:
    
    def __init__(self, ifVisualize=True):
        self.ifVisualize = ifVisualize
        self.img =None
        self.resultEvent =None
 
    #TODO ? Useful or not       
    def setNewFrame(self, newImg):
        self.img =newImg.copy()
 
 
    # TODO  profundly refactor ...       
    def visualize(self, resultTransfer: ResultTransfer, traject: Trajectory, 
                  faces, tm, score=None, select_idx =None ):
        filteredCenter = traject.smoothObs[-1][:2]   
        img = self.img   
        # --------  in detection -------------------------------------

        if traject.getMode() == 'detection':
     
            # Transfer the list of (index, faceName, certainty) in a given frame 
            index, faceName, certainty = resultTransfer.receive()    
            # TODO VERIFY frame index  ??? 
            img = v.visualizeDetection(img, faces, filteredCenter,select_idx, tm, faceName, certainty)

        # --------in tracking-------------------
        if traject.getMode() =='tracking':    # faces = faceTuple
            img = v.visualizeTracking(img, faces, filteredCenter, score, tm)


        img = v.visualizeTraject(img, traject.observations,GREEN)
        img = v.visualizeTraject(img, traject.smoothObs, BLUE)
        #cv.imshow('Video', img)
        #return img
        self.img = img


    