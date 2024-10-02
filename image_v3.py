
# -*- encoding: utf-8 -*-

import multiprocessing as mp

import cv2 as cv

from trajectory import Trajectory
import visualization as v
from visualization import GREEN, BLUE


class Image:
    
    def __init__(self, q: mp.Queue, ifVisualize=True):
        self.ifVisualize = ifVisualize
        self.img =None
        self.resultQueue = q # To receive the face recognition results in a synchronous way.
        
    def setNewFrame(self, newImg):
        self.img =newImg.copy()
        
    def visualize(self, traject: Trajectory, faces, tm, score=None, select_idx =None ):
        filteredCenter = traject.smoothObs[-1][:2]   
        
        img = self.img   
        # --------  in detection -------------------------------------
        if traject.getMode() == 'detection':
            index, faceName, certainty = self.resultQueue.get()
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

    def show(self):
        cv.imshow('Video', self.img)
