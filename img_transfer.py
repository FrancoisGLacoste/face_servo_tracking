# -*- encoding: utf-8 -*-

"""  
img_transfer.py

"""
#import multiprocessing as mp
from multiprocessing import Queue, shared_memory, Event

import numpy as np

# TODO:  use ContextManager to ensure the shared memory closes without issue even if the app crashes


class ImgTransfer:
    FRAME_NUMBER = 3              # Total number of frames to hold in shared memory
    
    def __init__(self, isOn=True):
        self.isOn = isOn          # i.e. image transfer is on  
        
        # For the image transfer to the face recognition task
        self.boxQueue = Queue()   # Queue for image box coordinates 
        self.imgQueue = Queue()   # Queue for image metadata
        
        # For the image transfer to imageDisplay
        self.boxQueue = Queue()   # Queue for image box coordinates 
        self.imgQueue = Queue()   # Queue for image metadata
        
        # For the image transfer to both ImageDisplay and face recognition task:
        self.ready_event = Event()# Signal when shared memory is ready for access
        self.done_event  = Event()    
        
        self.shm = None          # In the process of createShareMemory and shareImage
        self.frameSize = 0       # Set and used to create shm
        self.existing_shm =None  # In the process of retrieveImage and ImgDisplay
        
        self.frameIndex = 0      # Index of the image frame in the shared memory
        # When a value changes in a process, it does not change in another process unless it is send by queue

        
    def __del__(self):
        try:
            if self.shm is not None:
                self.shm.close()  
                self.shm.unlink()  
            if self.existing_shm is not None:
                self.existing_shm.close()
                self.existing_shm.unlink()  # Should not be needed if shm is unlinked
            print('Succesfully cleaned up the shared memory')    
        except Exception as e:
            print(f"Error cleaning up shared memory: {e}")
    
    def sharedMemoryExists(self, shmName):
        """   
        Test if a shared memory named shmName already exists in memory. 
        No simple way to implement.... 
        """     
        NotImplemented
           
    def createSharedMemory(self, frameSize):            
        frameNb =ImgTransfer.FRAME_NUMBER
        self.frameSize = frameSize
        name = 'image_transfer'
        
        #if self.shm is None and not self.sharedMemoryExists(name): 
        if self.shm is None :        
            # It is possible that the shared memory already exists 
            try:
                self.shm = shared_memory.SharedMemory(name=name)
                print(f"Loaded existing shared memory: {name}")
            except FileNotFoundError:
                # If it doesn't exist, create it
                self.shm = shared_memory.SharedMemory(create=True, 
                                                    size=frameSize*frameNb, 
                                                    name = name)
                print(f"Created new shared memory: {name}")
                
            except Exception as e:
                print(f'Cannot create the shared memory. Error: {e}')    
 
    # ================================================================================
    # On the sender side    
     
    def share(self, frame: np.array, faceBoxes: np.array, hasToRunRecognition: bool):
    
        self.shareImage(frame )

        try:      
            self.imgQueue.put((self.shm.name, frame.shape, frame.dtype, self.frameIndex) )
            print('Metadata have been put in the queue.')

            # Put face boxes in the queue
            self.boxQueue.put(faceBoxes[:,:4])
            
            self.ready_event.set()  # Tell retrieveImage it is ready
            self.done_event.wait()  # ? wait for confirmation from retrieve_image() AND ImgDisplay 
            self.done_event.clear()
            
        except Exception as e:
            print(f'Error: {e}')

    def shareImage(self, frame: np.array):
        """ 
        We dont need a lock to protect the memory writing because this is the only
        single task that accesses the shared memory for writing. 
        """
        print(f'We want to share a frame of imgSize= {frame.nbytes} bytes.')
        index = self.frameIndex   
        size  = self.frameSize
        try:
            # Write frame to shared memory at the calculated offset
            np_array = np.ndarray(frame.shape, dtype=np.uint8, 
                                    buffer = self.shm.buf[index*size: (index + 1) * size])
            np.copyto(np_array, frame)
            print(f'The data (frame) have been copied in the shared memory.')
            
            # Update index (circular buffer)
            self.frameIndex = (index + 1) % ImgTransfer.FRAME_NUMBER

        except MemoryError as e:
            print(f'Memory Error:{e}') 
        except Exception as e:
            print(f'Error in memory sharing: {e}')             
        except Exception as e: 
            print(e)    


    # =========================================================================================
    #  On the receiver side
    
    def updateEvents(self):             
        """Update the readers_done counter
        """
        Ntasks =2 # Number of different tasks that read the shared memory
        with self.lock:
            """The lock ensures that increments are thread-safe.
            ready_event is set when the frame is ready.
            done_event is set when all the Ntasks have finished reading. 
            
            (This lock has minimal impact on performance )
            """
            self.readers_done.value += 1
            if self.readers_done.value >= Ntasks:  
                self.done_event.set()
                self.readers_done.value = 0
        self.ready_event.clear()

    def retrieveImage(self):
        
        try:
            shm_name, shape, dtype, frameIndex = self.imgQueue.get()
            print('Image metadata have been retrieved from  imgQueue.')
                
            self.existing_shm = shared_memory.SharedMemory(name=shm_name)
            frameSize = self.existing_shm.size 
            buffer = self.existing_shm.buf[frameIndex*frameSize: (frameIndex + 1) * frameSize]
            np_array = np.ndarray(shape, dtype=dtype, buffer=buffer)
            image_copy = np.array(np_array)
            print(f'The image have been retrieved from the shared memory. type={type(image_copy)} ')
        
            return image_copy
        except Exception as e: 
            print(f'Error: {e}')
   

    def retrieve(self):
        
        try:
            self.ready_event.wait()  # wait for shareImage to be ready_event.set() 
            image_copy = self.retrieveImage()
            self.updateEvents()
        
            imgBoxes = self.boxQueue.get() # List of face box coordinates
            print('Face box coordinates have been retrieved from boxQueue')
    
        except Exception as e: 
            print(f'Error: {e}')
        
        
        