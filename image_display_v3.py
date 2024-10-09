
# -*- encoding: utf-8 -*-

"""  
image_display_v3.py
"""
import numpy as np
import cv2 as cv

from trajectory import Trajectory
from img_transfer import ImgTransfer
from result_transfer import ResultTransfer


import visualization_v3 as v
from visualization_v3 import GREEN, BLUE

# TODO : in this class: 
#        Using ImgTransfer retrieve the frame, face objects ( in dictionary or serialized form) 
#        This class is between ImgTransfer and  the tornado VideoStreamHandler
#        annotate the frames with the other informations to display, and transform them in jpg
#        It is used in the frame generator that serves VideoStreamHandler
            
class ImageDisplay:
    
    def __init__(self,imgTransfer: ImgTransfer, ifVisualize=True):
        self.ifVisualize = ifVisualize
        self.imgTransfer = imgTransfer
    
 
    async def prepareFrame(self):
        """  Returns annotated frame with boxes, observed centers, """
        image, facesInfos = self.imgTransfer.retrieveFaceInfos('display')  
        traject = self.imgTransfer.trajectQueue.get()
        return self.annotateFrame(image, facesInfos , traject)
    
    
    def annotateFrame(self, image: np.array, facesInfos: list , traject :Trajectory):
        """ 
        Annotate each camera frame with informations coming from face_detection or face_tracking
        through ImgTransfert.
          
        Called in server
        """
        
        return np.array([]) # TODO : A FAIRE
        
    '''#TODO ? Useful or not       
    def setNewFrame(self, newImg):
        self.img =newImg.copy()
    '''
 
    # TODO  rewrite entierely ...       
    '''
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

    '''    