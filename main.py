# -*- encoding: utf-8 -*-

import asyncio
import multiprocessing as mp

from face_recognition_SFace_oo import FaceRecognition
from camera_loop_oo import camera_loop
from camera_loop_oo import test_camera_loop_1, test_retrieve_results_1

async def main():   
    
    # Create a FaceRecognition object, including input/result queues
    faceRecognition = FaceRecognition() 
    faceRecognition.prepareFaceRecognition() # compute features, train the kNN classifier, save that.
       
    # Create the (CPU-bound) process for the servo-tracking of faces
    camera_process = mp.Process(target=camera_loop, 
                           args=(faceRecognition,) )
    camera_process.start()
     
    # During the face servo-tracking execution, those asynchronous loops are also running concurrently:
    await asyncio.gather(
            faceRecognition.runFaceRecognitionTask(), #face recognition runs on a separate thread pool.
            faceRecognition.retrieveResults(), # where recognition results are retrieved .
            faceRecognition.processNewFaceId(),# where unrecognized faces are processed.   
            )
    
    camera_process.join()
    
    
# =========================================================================================    
async def test_main_async_process():
    """ Test 1 
        To test the asyncio/process/threads structure
        Without doing anything but passing numbers as data
    test_face_servo_tracking_1 <-----> face_servo_tracking  
    test_retrieve_results_1    <-----> retrieve_results
    faceRecognition.test_runTask 
    """
    faceRecognition = FaceRecognition() 
    camera_process = mp.Process(target=test_camera_loop_1, 
                           args=(faceRecognition,))
    camera_process.start()
    await asyncio.gather(
            faceRecognition.test_runTask(),
            # injecting the queue could be a better practice 
            test_retrieve_results_1(faceRecognition.resultQueue)       
            )
    
    camera_process.join()    
         
if __name__ == '__main__':
    #asyncio.run(test_main_async_process())
 
    asyncio.run(main())
     