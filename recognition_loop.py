# -*- encoding: utf-8 -*-

"""
recognition_loop.py  
"""


import asyncio
from concurrent.futures import ProcessPoolExecutor  #ThreadPoolExecutor

from face_recognition_SFace_oo_v3 import FaceRecognition
from img_transfer import ImgTransfer
from result_transfer import ResultTransfer

def recognitionLoop():
    """   Run in process, and asynchronously c"""

    # Create a FaceRecognition object
    faceRecognition = FaceRecognition() 
    print('We just created the faceRecognition object in recognitionLoop.')
 
    #Launches a pool of processes for the task of face recognition: 
    asyncio.run(target=faceRecognitionTask, )
    # Asynchronous event-loops  :
    '''
    await asyncio.gather(
            faceRecognition.retrieveResults(), # where recognition results are retrieved .
            faceRecognition.processNewFaceId(),# where unrecognized faces are processed.   
            )
    '''
    
# ================== For the execution of face recognition loop ===============          
async def faceRecognitionTask(faceRecognition: FaceRecognition, imgTransfer: ImgTransfer, resultTransfer: ResultTransfer):
    """ 
    """
    
   
    while True:  
        # This loop mirrors the camera videocapture loop
        print('Ready to receive the last face image that has been detected.')
        img, boxes =imgTransfer.retrieveImage()   
        print('We got a new face image for the face recognition task to process.')
        
        # Face recognition per se 
        faceImgs = faceRecognition.cropBoxes(img, boxes)    
        results=list()
        for faceImg in faceImgs:            
            recognizedName, certainty = faceRecognition.recognizeFace(faceImg)
            results.append((faceImg, recognizedName, certainty))
            index = '?'
            resultTransfer.resutQueue.put([index, recognizedName, certainty])    
            # Transfer the results to imgDisplay via a mp.queue or via mp.Manager
            # TODO How to synchronize the display of recognizedName and certainty on the video frame ?
            # ? Est-ce je devrais pas numeroter les frames (un index) pour nous assurer de 
            # retourner les resultats a la bonne image ?
        
        '''# Put the result in the result queue: the result is sent to retrieveResults()
        await self.resultTransfer.put(results) 
        #TODO For now it is sent via a queue, but it should 
        # eventually be connected via TCP/IP to a remote client that runs a GUI for the user.  
        '''

