# -*- encoding: utf-8 -*-

import sys
import signal # for exiting gracefully by tapping CTRL-C, CTRL-Z
#import asyncio

import multiprocessing as mp
import logging 

#import tornado.ioloop

from img_transfer import ImgTransfer
from result_transfer import ResultTransfer
from camera_loop_oo_v3 import cameraLoop
from recognition_loop import recognitionLoop
#from face_recognition_SFace_oo_v3 import faceRecognitionTask
from server import Server

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
    # eventually: rewrite cameraLoop in C++
    cameraProcess = mp.Process(target=cameraLoop, args=(imgTransfer,) ) 
   
    # Create the process that run the faceRecognition task in an async event-loop,   
    recognitionProcess = mp.Process(target=recognitionLoop, args=(imgTransfer,resultTransfer,)) 
    
    
    # Tornado server for videoStreaming and GUI
    server = Server(imgTransfer, port=8888)
    serverThread = server.startInThread()
                                                                        
    cameraProcess.start()
    recognitionProcess.start()
    
    cameraProcess.join()
    recognitionProcess.join()
    serverThread.join() 
    
# ===============================================================================
#   TODO ?            Test

# ===============================================================================
if __name__ == '__main__':
    main()
    
  