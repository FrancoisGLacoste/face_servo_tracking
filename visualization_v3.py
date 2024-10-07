# -*- encoding: utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#from face_detection_yunet_oo import FaceDetection 
#from visual_tracking_oo import FaceTracking
#from face_recognition_SFace_oo import FaceRecognition
#from trajectory import Trajectory
#import visual_tracking_oo as vt


GREEN = (10,255,0)
BLUE  = (255,0,0)
BLACK = (0,0,0)
RED = (0,0,256)
YELLOW = (50,200,200)
MAGENTA=(255, 0, 255)
CYAN = (255,255,0)
BLACK = (0,0,0)


# =========================================================================
# TODO  :  BETTER ORGANIZE, CLEAN   etc 
#  An IDEA:  
#   the fonctions:   visualizeTraject_inDetection  &  visualizeTraject_inTracking 
#   could use some generic visualization functions that are defined in visualization.py 
#   but be defined as methods of a new class Trajectory
#  Objects faceDetection and faceTracking, and eventually faceRecognition would be passed as argument
# Better that defining visualizeDetection and visualizeTraject as it we currently do .


# ================  for filtered trajectories ===================================

       
def visualizeTraject(img, listOfPts, color=GREEN):
    """
    Draw the trajectory( listOfPts) of the non-filtered face center coordinates upon the video frame image,
    Args:
        img :_type_: image, frame from the video
        faceCenterTraject :list of (int16,int16) coordinates: 
                                Trajectory of the non-filtered face center coordinates

    Returns:
        _type_: image from the video with an additional trajectory drawn upon it
    """
    try:
        for p in listOfPts:
            x = int(np.round(p[0]))
            y = int(np.round(p[1]))
            cv.circle(img, (x,y), 1, color, 1)
    #except # just skip if fails
    
    finally:    
        return img


# =============     For Face Detection ============================================
'''def visualizeTraject_inDetection(faceCenterTraject,filteredTraject,#detectionTraject: Trajectory,
                                 img,faces, select_idx, tm):
    # Rem: output img can be written into a video (videoWrite) 
    #faceCenterTraject = detectionTraject.observations
    #smoothCenterraj = detectionTraject.filteredObs    # list of tuples
    img = visualizeDetection(img, faces, filteredTraject[-1][:2],select_idx, tm)
    img = visualizeTraject(img, faceCenterTraject, GREEN)
    img = visualizeTraject(img, filteredTraject, BLUE)
    #cv.imshow('Video', img)
    return img
'''

def visualizeDetection(img, faceArray,faceCenter, select_id, tm, faceName, certainty,verbose=False ):
    
    """
    We show the face center only for the selected face (select_id). 
    The selected face is emphasized: it is the one that is tracked by the servos
    
    """
    print((faceName, certainty))   # TODO A AJOUTER sur la video
    
    fps =  tm.getFPS()  # apriori not the same as cv.CAP_PROP_FPS
    detectTime = tm.getTimeMilli()
    #avgDetectTime = tm.getAvgTimeMilli()  # when tm is not reset 

    thickness=2 
    selectFace_thick = 4
    selectFace_color = GREEN


    if faceArray is not None: 
        x_center, y_center = faceCenter.reshape(2,) # array (2,1)
        
        faceNb = np.size(faceArray,0) 
        if verbose:
            print(f'I am seeing {faceNb} face(s) !')
    
        for idx, face in enumerate(faceArray):
            score = face[-1]
            coords = face[:-1].astype(np.int32)              
            x,y,w,h,x_eye1,y_eye1,x_eye2,y_eye2,x_nose,y_nose,x_mouth1,y_mouth1,x_mouth2,y_mouth2 = coords
            message = '''Face {}: nose = ({:.0f}, {:.0f}),
                        eye1 = ({:.0f}, {:.0f}),eye2 = ({:.0f}, {:.0f}), 
                        surface = {:.0f}, score = {:.2f}'''                     
            if verbose:
                print(message.format(idx,x_nose, y_nose,x_eye1,y_eye1,x_eye2,y_eye2, w*h, score))
            
            boxColor = BLACK
            boxThickness = thickness
            captionText ='.'
            
            # Specific to the selected face
            if idx == select_id: 
                # The face center is displayed only for the selected face
                #print(type(x_center))  # np.int16
                
                captionText = f", center = {x_center}, {y_center}"
                #captionText = ', center = ({:.0f}, {:.0f}).'.format(x_center, y_center)
                cv.circle(img, (x_center, y_center), 4, selectFace_color, selectFace_thick)
                boxThickness = selectFace_thick
                boxColor = selectFace_color
            
            cv.rectangle(img, (x, y), (x+w, y+h),boxColor, boxThickness)
            cv.circle(img, (x_center, y_center), RED, thickness)
            '''cv.circle(img, (x_eye1,y_eye1), 2, RED, thickness)
            cv.circle(img, (x_eye2,y_eye2), 2, BLUE, thickness)
            cv.circle(img, (x_nose,y_nose), 2, CYAN, thickness)
            cv.circle(img, (x_mouth1,y_mouth1), 2, MAGENTA, thickness)
            cv.circle(img, (x_mouth2,y_mouth2), 2, YELLOW, thickness)            
            '''
            try:
                cv.putText(img,
                        ('Face {}: surface = {:.0f}, score = {:.2f}'
                            .format(idx, w*h, score) + captionText),
                        (1, 15*(2+idx)), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        0.5, boxColor, 2)
            except Exception as e: 
                print(' Visualization of face detection has met an error...') 
                print(e)      
        
    # show the image even when no face is detected               
    cv.putText(img, 'Frames per second: {:.2f}; Detection time: {:.2f} ms'.format(
        fps, detectTime), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2) 
    return img



# ================ For Face Tracking =====================================================
'''def visualizeTraject_inTracking(faceCenterTraject, filteredTraject,
                                img,faceTuple,score, tm):

    #faceCenterTraject = trackingTraject.observations
    #filteredTraject = trackingTraject.filteredObs
    smoothCenter = filteredTraject[-1]
    img = visualizeTracking(img, faceTuple, smoothCenter[:2], score, tm)
    img = visualizeTraject(img, faceCenterTraject, GREEN)
    img = visualizeTraject(img, filteredTraject, BLUE)
    #cv.imshow('Video', img)
    return img
 '''  
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

    
# ==================================================================

def plotTraject(x,y, t, x_label = "t     [seconds]" ):
    fig, axs = plt.subplots(2)#(figsize=(10, 6))
    fig.suptitle('Fitting face trajectory in tracking mode')
    #axs[0].plot(t, x)
    #axs[1].plot(t, y)
    #ax.set_title("Kalman Filter in 2D Motion Estimation")
    
    for n,z in enumerate([x,y]):
        axs[n].scatter(t, z, c='red', s=1, label=f"Noisy Observations in {z}")
    #ax.plot(predictions[:, 0], predictions[:, 1], 'b-', label="Kalman Filter Estimation")
    axs[1].set_xlabel(x_label)
    #ax.set_ylabel("x")
    #ax.set_xlim(0, 10)
    #ax.set_ylim(-1.5, 1.5)
    #ax.legend()


    #----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Fitting face trajectory in tracking mode')
    #ax.plot(t, x)   
    ax.scatter(t, x, c='red', s=1, label=f"Noisy Observations in x")
    ax.set_xlabel(x_label)
    #----

    plt.show() 

# -------------
def showTraj(measurements, predictions):
    measurements = np.array(measurements).squeeze()
    predictions = np.array(predictions)
    predictions = predictions.squeeze()

    # sample number
    nb =predictions[:, 0].shape[0]  
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 700)
    ax.set_ylim(250, 450)
    ax.set_title("Kalman Filter in 2D Motion Estimation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
                
    # Plotting the noisy measurements and Kalman filter estimations
    ax.scatter(measurements[:, 0], 
                measurements[:, 1], 
                c='red', s=10, label="Noisy Measurements")
    ax.plot(predictions[:, 0].reshape((nb,1)), 
            predictions[:, 1].reshape((nb,1)),
            'b-', label="Kalman Filter Estimation")
    
    #ax.scatter(predictions[:, 0].reshape((nb,1)), 
    #    predictions[:, 1].reshape((nb,1)), 
    #    c='blue', s=10, label="Kalman Filter Estimation")

    ax.legend()        
    plt.show() 


    