# -*- encoding: utf-8 -*-

import sys
import signal # for exiting gracefully by tapping CTRL-C, CTRL-Z
#import asyncio
import threading
import multiprocessing as mp
#import logging 

#import tornado.ioloop

from img_transfer import ImgTransfer
from result_transfer import ResultTransfer
from camera_loop_oo_v3 import cameraLoop
from recognition_loop import recognitionLoop
#from face_recognition_SFace_oo_v3 import faceRecognitionTask
from server import Server
from monitoring import monitoringLoop

def handle_exit(signum, processes:list):
    """To exit 'gracefully' when tapping CTRL-C , CTRL-Z etc..."""
    print("Process interrupted through CTRL-Z . Joining processes...")
    for p in processes: 
        p.join()
    sys.exit(0)
         
def main(): 
   
    imgTransfer = ImgTransfer()  # Transfers the detected face to the recognition module
    resultTransfer = ResultTransfer()     # Sends the result to the server
    
    # Create the (CPU-bound) process for the servo-tracking of faces
    # TODO eventually: rewrite cameraLoop in C++ and divide ImgTransfer class in 2: sender vs rcver
    cameraProcess = mp.Process(target=cameraLoop, args=(imgTransfer,) ) 
   
    # Create the process that run the faceRecognition task : This task itself runs 
    # into an async event-loop that runs inside the recognitionProcess   
    recognitionProcess = mp.Process(target=recognitionLoop, args=(imgTransfer,resultTransfer,)) 
                                                                        
    cameraProcess.start()
    recognitionProcess.start()

    # Register 
    signal.signal(signal.SIGINT, handle_exit)   
    signal.signal(signal.SIGTERM, handle_exit)  # To handle exit signals like ctrl-Z 
      
    """ 
    # EXPLANATION: 
    (I) We want that: 
    (1) The two processes  (cameraProcess and recognitionProcess) and the Tornado server 
        must run simultaneously during normal operation condition.
    (2) If the server is interrupted, the two processes must continue to run 
                                                (i.e. not being interrupted)
    (3) If one of the two processes is interrupted (crash or otherwise), the other process must also 
        be interrupted as well as the Tornado server. 
    
    (II) Moreover: we know that:
    (1) Running the Server into a separate thread can lead to unexpected crash of the server
        ( in particular when the server uses other async libraries such as aoifiles or async-files.
    (2) It is advised ( in Tornado doc) to not run the Tornado async event-loop in a separate thread. 
        The event loop is not thread-safe.
    
    (3) Consequently: the Tornado IOLoop must run directly in the program main. 
        It blocks: anything after this event-loop will never be executed 
        unless the server is interrupted. 
    
    (4) Consequently, we cannot join the processes AFTER the Tornado IOLoop.
    
    (5) If processes are not joined when a they are interrupted, then they become zombie processes. 
     
    
    (6) Consequently: we should provide a mechanism to monitor the two processes and join them in 
    case of unexpected interruption ( and close them gracefully).

    (6.1) One way is to use tornado.ioloop.IOLoop.current().call_later method, to call a 
    new function "monitor_processes"  that monitor proc1, proc2 and gracefully close 
        ex: tornado.ioloop.IOLoop.current().call_later(delay, monitor_processes, proc1, proc2)

        But it would be an unforgivable violation of the separation of concerns principle  
    (6.2) An alternative is to use an additional thread to run the 'monitor_processes' function.  

    (7) Remark: starting the processes in a try-except block will not allow to capture 
        a process that crashes after it has been started: 
        
    Consequently:  We chose to start a separate monitorThread (6.2 above)    
    """ 
    monitoringThread = threading.Thread(target=monitoringLoop, 
                                         args=([cameraProcess, recognitionProcess]))
    monitoringThread.start() 
    
    # Tornado server for videoStreaming and GUI
    server = Server(imgTransfer, port=8888)
    
    
# ===============================================================================
#   TODO ?            Test

# ===============================================================================
if __name__ == '__main__':
    main()
    
  