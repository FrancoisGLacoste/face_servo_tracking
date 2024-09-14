

import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from face_recognition_SFace_oo import FaceRecognition
from face_servo_tracking_oo import face_servo_tracking
from face_servo_tracking_oo import test_face_servo_tracking_1, test_retrieve_results_1

async def main():   
    
    # Create a FaceRecognition object, including input/result queues
    faceRecognition = FaceRecognition() 
    faceRecognition.prepareFaceRecognition() # compute features, train the kNN classifier, save that.
       
    # Create the (CPU-bound) process for the servo-tracking of faces
    camera_process = mp.Process(target=face_servo_tracking, 
                           args=(faceRecognition,) )
    camera_process.start()
     
    # During the face servo-tracking, face recognition is performed on a separate thread pool
    await asyncio.gather(
            faceRecognition.runFaceRecognitionTask(),
            faceRecognition.retrieveResults(),  # where results are retrieved 
            faceRecognition.retrieveNewFaceId(),   
            )
    
    camera_process.join()
    
    
    
async def test_main_async_process():
    """ Test 1 
        To test the asyncio/process/threads structure
        Without doing anything but passing numbers as data
    test_face_servo_tracking_1 <----- face_servo_tracking  
    test_retrieve_results_1    <----- retrieve_results
    faceRecognition.test_runTask 
    """
    faceRecognition = FaceRecognition() 
    camera_process = mp.Process(target=test_face_servo_tracking_1, 
                           args=(faceRecognition,))
    camera_process.start()
    await asyncio.gather(
            faceRecognition.test_runTask(),
            test_retrieve_results_1(faceRecognition.resultQueue)       
            )
    
    camera_process.join()    
         
if __name__ == '__main__':
    #asyncio.run(test_main_async_process())
 
    asyncio.run(main())
     