# -*- encoding: utf-8 -*

"""
visual_tracking.py
"""
import os 

import numpy as np
import cv2 as cv

import file as fl
GREEN = (10,255,0)
BLUE = (255,0,0)
RED =  (0,0,255) 
YELLOW = (50,200,200)
MAGENTA=(255, 0, 255)
CYAN = (255,255,0)
BLACK = (0,0,0)

#fl.BASE_DIR = '/home/moi1/Documents/dev_py/vision/PROJET_Face-Tracking-camera/'

def isCuda():  
    try:
        return (cv.cuda.getCudaEnabledDeviceCount() > 0)
       
    except:
        return False


#=====================================================================
#       Tracking with vittrack
#    https://docs.opencv.org/4.x/d9/d26/classcv_1_1TrackerVit.html
#    https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
#=====================================================================

class FaceTracking():
    def __init__(self):
        self.tracker = self.createFaceTracker_vittrack()
        
    def createFaceTracker_vittrack(self):   
        
        if isCuda():
            backend = 1 # i.e. cv.dnn.DNN_BACKEND_CUDA
            target  = 1 # i.e. cv.dnn.DNN_TARGET_CUDA

        else :
            backend = 0 # i.e. cv.dnn.DNN_BACKEND_OPENCV
            target  = 0 # i.e. cv.dnn.DNN_TARGET_CPU
        
        param = cv.TrackerVit.Params()
        param.net     = os.path.join(fl.BASE_DIR, 'object_tracking_vittrack_2023sep.onnx')
        param.backend = backend
        param.target  = target 
        tracker = cv.TrackerVit.create(param)
        return tracker
    
    
    def initTracker(self, img, face):
        """
        face:     np.ndarray of shape (4,)  :  face to track
        """ 
        self.tracker.init(img,tuple(face))     
        
def visualizeTracking(image, bbox, smoothCenter, score, tm, name=None):
   
    
    fontScale = 1
    fontSize  = 0.5
    thickness = 2
    textColor =GREEN
    output = image.copy()
    h, w, _ = output.shape
    
    fps =  tm.getFPS()  #  not the same as cv.CAP_PROP_FPS
    detectTime = tm.getTimeMilli()
 
    
    if name is not None:
        cv.putText(output, f'Id: {name}', (0, 60), cv.FONT_HERSHEY_SIMPLEX, fontScale, GREEN, fontSize)
    
    #cv.putText(output, 'Frames per second: {:.2f}'.format(fps), (0, 30), cv.FONT_HERSHEY_DUPLEX, fontScale, textColor, fontSize)
    cv.putText(output, 'Frames per second: {:.2f}; Detection time: {:.2f} ms'.format(fps, detectTime), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1) 
 

    # bbox: Tuple of length 4
    x, y, w, h = bbox
    #print(type(x))
    cv.rectangle(output, (x, y), (x+w, y+h), GREEN, 2)
  
    xc, yc =smoothCenter.reshape(2,).astype(int) 
    #print(smoothCenter)
    #print(xc)
    #print(type(xc))
    
    captionText = ', center = ({:.0f}, {:.0f}).'.format(xc, yc)
    #cv.circle(output, (int(xc), int(yc)), 4, GREEN, thickness)
    cv.circle(output, (xc, yc), 4, GREEN, thickness)
    cv.putText(output, '{:.2f}'.format(score), (x, y+25), cv.FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness)

    '''
    text_size, baseline = cv.getTextSize('Target lost!', cv.FONT_HERSHEY_SIMPLEX, fontScale, fontSize)
    text_x = int((w - text_size[0]) / 2)
    text_y = int((h - text_size[1]) / 2)
    cv.putText(output, 'Target lost!', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), fontSize)
    '''
    return output

    

