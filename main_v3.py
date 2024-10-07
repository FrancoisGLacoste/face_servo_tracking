# -*- encoding: utf-8 -*-

import sys
import signal # for exiting gracefully by tapping CTRL-C, CTRL-Z
#import asyncio
import multiprocessing as mp
import logging 

import numpy as np

from img_transfer import ImgTransfer
from result_transfer import ResultTransfer
from camera_loop_oo_v3 import cameraLoop
from recognition_loop import recognitionLoop
#from face_recognition_SFace_oo_v3 import faceRecognitionTask

def handle_exit(signum, frame):
        print("To exit 'gracefully' when tapping CTRL-C , CTRL-Z etc...")
        sys.exit(0)
     
def main(): 
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    signal.signal(signal.SIGINT, handle_exit)   
    signal.signal(signal.SIGTERM, handle_exit)  # To handle exit signals like ctrl-Z 

    imgTransfer = ImgTransfer()  # Transfers the detected face to the recognition module
    resultTransfer = ResultTransfer()     # Sends the result to the server
    
    # Create the (CPU-bound) process for the servo-tracking of faces
    camera_process = mp.Process(target=cameraLoop, args=(imgTransfer,resultTransfer,) )
   
    # Create the process that run the faceRecognition task in an async event-loop,   
    recognition_process = mp.Process(target=recognitionLoop, args=(imgTransfer,
                                                                    resultTransfer,)  ) 
    camera_process.start()
    recognition_process.start()

     

    camera_process.join()
    recognition_process.join()
    
    
# ===============================================================================
#   TODO ?            Test

# ===============================================================================
if __name__ == '__main__':
    main()
    
  