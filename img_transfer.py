# -*- encoding: utf-8 -*-

"""  
img_transfer.py

"""
#import multiprocessing as mp
from multiprocessing import Queue, shared_memory, Event

import numpy as np

from faces import Face
from trajectory import Trajectory

# TODO:  use ContextManager to ensure the shared memory closes without issue even if the app crashes


class ImgTransfer:
    FRAME_NUMBER = 3              # Total number of frames to hold in shared memory
    
    def __init__(self, isOn=True):
        self.isOn = isOn          # i.e. image transfer is on  
        
        # Queue for image metadata needed to access shared memory
        self.imgQueue = {'recognition': Queue(),   # Image transfer to the face recognition task
                         'display':     Queue()    # Image transfer to the video stream
                         }
       
        self.facesQueue = {
            'recognition': Queue(), # for sending box coordinates to the face recognition task
            'display':     Queue()  # for sending face dict infos to imageDisplay and the video stream 
            }
        
        self.trajectQueue = Queue() # for sending trajectories to imageDisplay and the video stream 
         
        # For the image transfer to both ImageDisplay and face recognition task:
        self.ready_event = Event()# Signal when shared memory is ready for access
        self.done_event  = Event()  # TODO Not sure when I will use it finally *********  
        
        self.shm = None          # In the process of createShareMemory and shareImage
        self.frameSize = 0       # Set and used to create shm
        self.existing_shm =None  # In the process of retrieveImage and ImgDisplay
        
        self.frameIndex = 0      # Index of the image frame in the shared memory
        # When a value changes in a process, it does not change in another process unless
        # it is send by queue

        
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
     
    def shareFaces(self, frame: np.array, faces: list, hasToRunRecognition: bool):
        """  
        Args:
            frame : image frame 
            faces : list of Face objects 
            hasToRunRecognition : if we also send data to the recognition module.
        """
        self.shareImage(frame )

        # Face informations we need to send to the recognition loop and the image display 
        faceInfos = {'recognition': [face.boxes  for face in faces],
                        'display'    : [face.dict() for face in faces]}

        targets = ['recognition','display'] if hasToRunRecognition else ['display']
        try:          
            for target, content in zip(targets, faceInfos):
                # Sending face information to the target module
                self.facesQueue[target].put(content)   

                # Sending image metadata to the target module
                self.imgQueue[target].put((self.shm.name, frame.shape, 
                                                frame.dtype, self.frameIndex) )
                print('Image metadata have been put in the image queue.')
                self.ready_event.set()  # Tell retrieveImage it is ready

                '''
                # Once an image is shared, it does not prevent it to write the next image
                # even if some processses are still reading the previous image. 
                # Because we write and read in a shared memory that can contains at least 3 images.
                
                TODO : What to do with these events ???  Is seems useful but no more sure.... 
                self.done_event.wait()  # ? wait for confirmation from retrieve_image() AND ImgDisplay 
                self.done_event.clear()
                '''
        except Exception as e:
            print(f'Error: {e}')

    def shareImage(self, frame: np.array):
        """ 
        Put an image (frame) in the shared memory block at the position of frameIndex.
        
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

    def sendTraject(self,traject: Trajectory):
        
        # TODO : add a condition: do we really always want to display the trajectories
        # It is only interesting with respect to the kalman filter calibration, 
        # But it is not for the final typical user.  
        self.trajectQueue.put(traject)
        
        
    # =========================================================================================
    #  On the receiver side
    
    def hasBeenRetrievedTwice(self):             
        """
        When the image has been retrieved twice  
        ( i.e. when recognition loop and the server have both retrieved the image), 
        then it signals the done event. 
        
        ------------------------------
        TODO Attention, shareImage does not need to wait after this done_event before sharing 
        the next image
        ------------------------------------------------------------------------ 
        """
        Ntasks =2 # Two different tasks read the shared memory
        with self.lock:
            """The lock ensures that increments are thread-safe.
            ready_event is set when the frame is ready.
            done_event is set when all the Ntasks have finished reading. 
            
            (This lock has minimal impact on performance )
            """
            # Increment every time the image is retrieved
            self.readers_done.value += 1
            if self.readers_done.value >= Ntasks:  
                self.done_event.set()
                self.readers_done.value = 0
        self.ready_event.clear()

    def retrieveImage(self, target:str):
        """   
        Retrieves an image (frame) from the shared memory. 
        ( Does not retrieve the face boxes and other face imformations)
        
        target : string: target module, i.e. either 'recognition' or 'display'
        """
        try:
            shm_name, shape, dtype, frameIndex = self.imgQueue[target].get()
            print('Image metadata have been retrieved from  imgQueue.')
                
            self.existing_shm = shared_memory.SharedMemory(name=shm_name)
            frameSize = self.existing_shm.size 
            buffer = self.existing_shm.buf[frameIndex*frameSize: (frameIndex + 1) * frameSize]
            np_array = np.ndarray(shape, dtype=dtype, buffer=buffer)
            image = np.array(np_array)
            print(f'The image have been retrieved from the shared memory. type={type(image)} ')
        
            # increment to show that this function has been called once more time
            self.hasBeenRetrievedTwice()  # TODO  Not sure how to use the done_event.wait !!  
            
            return image
        except Exception as e: 
            print(f'Error: {e}')
   
    '''
    def retrieveFaceBoxes(self):
        """ Retrieve the image boxes to send them to the recognition loop"""
        try:
            self.ready_event.wait()        # wait for shareImage to be ready_event.set() 
            image = self.retrieveImage()
            imgBoxes = self.facesQueue.get() # Retrieve a list of all face boxes of a same frame 
            print('Face box coordinates have been retrieved from boxesQueue')
            return image, imgBoxes
        except Exception as e: 
            print(f'Error: {e}')
    '''    

    def retrieveFaceInfos(self, target: str):
        """ Retrieve the face informations to send them to the target module."""
        try:
            self.ready_event.wait()        # wait for shareImage to be ready_event.set() 
            image = self.retrieveImage(target)
            facesList = self.facesQueue[target].get() # Either list of face.dict() or of face.box
            print('Face dictionary informations have been retrieved from facesQueue')

            return image, facesList
        except Exception as e: 
            print(f'Error: {e}')
        
        