# -*- encoding: utf-8 -*-

import asyncio
import multiprocessing as mp

import numpy as np

from img_transfer import ImgTransfer
from camera_loop_oo_v3 import camera_loop
from face_recognition_SFace_oo_v3 import faceRecognitionLoop


async def main():   
    
    imgTransfer = ImgTransfer()
    resultQueue = mp.Queue()  
    # Create the (CPU-bound) process for the servo-tracking of faces
    camera_process = mp.Process(target=camera_loop, args=(imgTransfer,resultQueue,) )
    camera_process.start()
   
    # Create the process that run the faceRecognition task,   
    recognition_process = mp.Process(target=faceRecognitionLoop, args=(imgTransfer,resultQueue,)  ) 
    recognition_process.start()
   
     
    '''# Those asynchronous loops are also running concurrently:
    await asyncio.gather(
            faceRecognition.retrieveResults(), # where recognition results are retrieved .
            faceRecognition.processNewFaceId(),# where unrecognized faces are processed.   
            )
    '''
    camera_process.join()
    recognition_process.join()
    
    
# ===============================================================================
    def produceMockData():
        print('Producing data.')
        return np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

    def shareImg_process(transfer_obj: ImgTransfer, image_data):
        print('Enter into shareImg_process.')
        transfer_obj.shareImage(image_data)

    def retrieveImg_process(transfer_obj: ImgTransfer):
        print('Enter into retrieveImg_process.')
        transfer_obj.retrieveImage()

    def test_main():
        imgEx = produceMockData()
        imgSize =imgEx.size #3145728
        img_transfer = ImgTransfer(imgSize)

        # Image production process
        image_data = produceMockData()
  
        p1 = mp.Process(target=shareImg_process, args=(img_transfer, image_data))
        p2 = mp.Process(target=retrieveImg_process, args=(img_transfer,))

        p1.start()
        p2.start()
        p1.join()
        p2.join()   

if __name__ == '__main__':
 
    # test_main()    # Should I test with pytest ?
    asyncio.run(main())
     