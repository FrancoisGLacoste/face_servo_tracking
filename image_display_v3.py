
# -*- encoding: utf-8 -*-

"""  
image_display_v3.py
"""
import numpy as np
import cv2 as cv

from trajectory import Trajectory
from img_transfer import ImgTransfer


GREEN = (10,255,0)
BLUE  = (255,0,0)
BLACK = (0,0,0)
RED = (0,0,256)
YELLOW = (50,200,200)
MAGENTA=(255, 0, 255)
CYAN = (255,255,0)
BLACK = (0,0,0)



#        Using ImgTransfer retrieve the frame, face objects ( in dictionary or serialized form) 
#        This class is between ImgTransfer and  the tornado VideoStreamHandler
#        annotate the frames with the other informations to display, and transform them in jpg
#        It is used in the frame generator that serves VideoStreamHandler
            
class ImageDisplay:
    
    def __init__(self,imgTransfer: ImgTransfer, ifVisualize=True):
        self.ifVisualize = ifVisualize   # TODO WHERE WILL WE USE IT ?????
        self.imgTransfer = imgTransfer
    
 
    async def prepareFrame(self):
        """  
        Prepares annotated frame with boxes, centers, trajectories, and other infos 
        and returns it in JPEG format 
        """
        frame, facesInfos = self.imgTransfer.retrieveFaceInfos('display')  
        traject = self.imgTransfer.trajectQueue.get()
        
        frameWithBox  = self.displayBox(frame, facesInfos)
        annotatedFrame = self.annotateFrame(frameWithBox, facesInfos)
        finalFrame = self.displayTraject(annotatedFrame, traject)
        jpgImg = self.convertImgToJpg(finalFrame)
        return jpgImg
        
    def displayBox(self, frame: np.array, facesInfos: dict):
        """     
        """
        for faceId, faceInfo in enumerate(facesInfos):
            # faceId : face identifier
            x,y,w,h = faceInfo['box']
            #x_center, y_center = faceInfo['observedCenter']
            
            color = GREEN if faceId==1 else BLACK
            thickness = 2 if faceId==1 else 1
            cv.rectangle(frame, (x, y), (x+w, y+h),color, thickness)
            cv.circle(frame, faceInfo['observedCenter'], color, thickness)
            cv.circle(frame, faceInfo['smoothCenter'], BLUE, thickness)
            
            # Write the face identifier number in the corner of each face box
            try:
                cv.putText( frame,
                            faceId,
                            (x, y), 
                            cv.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            color, 
                            thickness ) 
            except Exception as e: 
                print(e)      

            return frame
        
    def annotateFrame(self, frame: np.array, facesInfos: dict ):
        """ 
        Annotate each camera frame with informations coming from face_detection or face_tracking
        through ImgTransfert.
          
        Called in server
        """
        for faceId, faceInfo in enumerate(facesInfos):
            mode = faceInfo['mode']  # either 'detection of 'tracking'
            try:
                color = GREEN if faceId==1 else BLACK
                thickness = 2 if faceId==1 else 1
                cv.putText(frame, 
                           f"""Face #{faceId}: Frames per second: {faceInfo['fps']:.2f}; 
                               {mode} time: {faceInfo['time']:.2f} ms; 
                               score: {faceInfo['score']}.""", 
                           (1, 15*(2+faceId)), 
                           cv.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, thickness) 
            except Exception as e: 
                #print(e)
                continue           
        return frame
                
       
    def displayTraject(self, frame: np.array, traj :Trajectory):
        """ 
        Draw the observed and filtered trajectories of face centers on the image frame.
        For Either the detection or tracking mode, 
    
        Args:
            traject : Either detection of tracking trajectory    
        Returns:
            frame: np.array : image from the video with the additional trajectories drawn upon it
        """
        try:
            if traj is not None:
                for listOfPts, c in zip([traj.observations, traj.smoothObs],[GREEN,BLUE]): 
                        for p in listOfPts:
                            x = int(np.round(p[0]))
                            y = int(np.round(p[1]))
                            cv.circle(frame, (x,y), 1, c, 1)
        except Exception as e:
            print('Error: {e}.')            
        finally:    
            return frame
 
        
    def convertImgToJpg(self, img):
        """Convert the image to JPEG format"""
        try:
            success, encoded_image = cv.imencode('.jpg', img)
            if success:
                return encoded_image.tobytes() 
            else:
                print("Failed to encode image.")
        except Exception as e:
            print("Error, Failed to encode image: {e}")                

