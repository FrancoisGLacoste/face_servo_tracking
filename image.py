
# -*- encoding: utf-8 -*-

from trajectory import Trajectory
import visualization as v
from visualization import GREEN, BLUE

import cv2 as cv

class Image:
    
    def __init__(self, ifVisualize=True):
        self.ifVisualize = ifVisualize
        self.img =None

    def setNewFrame(self, newImg):
        self.img =newImg.copy()
        
    def visualize(self, traject: Trajectory, faces, tm, score=None, select_idx =None ):
        filteredCenter = traject.filteredObs[-1][:2]   
        
        img = self.img   
        # --------  in detection -------------------------------------
        if traject.getMode() == 'detection':
            img = v.visualizeDetection(img, faces, filteredCenter,select_idx, tm)

        # --------in tracking-------------------
        if traject.getMode =='tracking':    # faces = faceTuple
            img = v.visualizeTracking(img, faces, filteredCenter, score, tm)


        img = v.visualizeTraject(img, traject.observations,GREEN)
        img = v.visualizeTraject(img, traject.filteredObs, BLUE)
        #cv.imshow('Video', img)
        #return img
        self.img = img

    def show(self):
        cv.imshow('Video', self.img)
