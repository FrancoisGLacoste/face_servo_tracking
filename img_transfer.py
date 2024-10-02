# -*- encoding: utf-8 -*-

"""  
img_transfer.py

"""

#To transfer image: either mp.queue or mp.shared_memory 
#import multiprocessing as mp
from multiprocessing import Queue, shared_memory, Event

import numpy as np

class ImgTransfer:
    def __init__(self):
        self.boxQueue = Queue()   # Queue for image box coordinates 
        self.imgQueue = Queue()   # Queue for image metadata
        
        self.ready_event = Event()# Lock that protects access to shared memory
        self.done_event  = Event()    
        
        self._shm = None         # In the process of createShareMemory and shareImage
        self._existing_shm =None # In the process of retrieveImage
            
    def createShareMemory(self, imgSize=None):
        try:
            #imgSize #image_data.nbytes
            
            self._shm = shared_memory.SharedMemory(create=True, size=imgSize)
        except Exception as e:
            print(e)        
        
    def shareImage(self, img):
        print('img size=',img.nbytes )#3145728
        try:     
            '''    
            if self.imgSize is None: 
                 self.imgSize = image_data.nbytes
            
            # here we create a new shared memory every time we share an image... 
            shm = shared_memory.SharedMemory(create=True, size=self.imgSize)
            '''
            np_array = np.ndarray(img.shape, 
                                dtype=img.dtype, 
                                buffer=self._shm.buf)
            np.copyto(np_array, img)
            print(f'The data (img) have been copied in the shared memory.')
            self.imgQueue.put((self._shm.name, img.shape, img.dtype))
            print('Metadata have been put in the queue.')
            
            self.ready_event.set()
            self.done_event.wait()  # wait for confirmation from retrieve_image()
            self.done_event.clear()
            self._shm.unlink()
            #  Unlinking should be done once you're sure that no other process will 
            #  access the shared memory. 
            #  Typically, this is done in the process that created the shared memory 
            #  after it is no longer needed by any process.'''
        except Exception as e: 
            print(e)    

    def retrieveImage(self):
        self.ready_event.wait()
        shm_name, shape, dtype = self.imgQueue.get()
        print('Image metadata have been retrieved from  imgQueue.')
        
        imgBoxes = self.boxQueue.get() # List of face box coordinates
        print('Face box coordinates have been retrieved from boxQueue')
        
        self._existing_shm = shared_memory.SharedMemory(name=shm_name)
        np_array = np.ndarray(shape, dtype=dtype, buffer=self._existing_shm.buf_)
        image_copy = np.array(np_array)
        print(f'The image have been retrieved from the shared memory. type={type(image_copy)} ')
        
        # Clean up
        self._existing_shm.close()  # Releases the memory associated with the shared memory object 
        #existing_shm.unlink()  # Unlink the shared memory to avoid memory leaks ( =/= close)
        # BETTER: unlinking is done at the end, before closing the program
        
        self.done_event.set() # TODO  PROBLEM ????*********
        # warnings.warn('resource_tracker: There appear to be %d '
        
        self.ready_event.clear()
        
        return image_copy, imgBoxes

    def cleanUp(self):
        """ Unlink the shared memory to avoid memory leaks """
        self._existing_shm.unlink()
        
        # doit on aussi faire
        self._shm.unlink()
        
        
        