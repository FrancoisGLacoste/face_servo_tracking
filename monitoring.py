# -*- encoding: utf-8 -*-

import sys    
import time 
from multiprocessing import Process # is_alive, join
   
def monitoringLoop(camera_process:Process, recognition_process:Process):
    while True:
        if not camera_process.is_alive() or not recognition_process.is_alive():
            print("One of the processes has exited.")
            print("We stop and join the other process.")
            if camera_process.is_alive():
                camera_process.join()
            if recognition_process.is_alive():
                recognition_process.join()
            print("All processes have been joined. Exiting.")
            sys.exit(0)  # Exit the program gracefully
        time.sleep(1)  # TODO Adjust as needed for performance

