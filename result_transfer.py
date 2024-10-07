# -*- encoding: utf-8 -*-


"""   
result_transfer.py

"""
import multiprocessing as mp # Queue, Event  (TODO: what about Manager instead ?)



class ResultTransfer:
    
    def __init__(self):
        self.timeout=0.08 # 80 ms 
        self.resultQueue = mp.Queue()
        self.resultEvent = mp.Event()
           
    def receiveResult(self):
        #resultEvent.wait()
        
        # It is said this kind of test is safer and idiomatic in Python
        try:
            # Wait 0.05 s or until a result arrives. 
            result = self.resultQueue.get(timeout=self.timeout)
            print(f'Received {result} for visualization.')
            index, faceName, certainty = result
        except TimeoutError : #resultQueue.Empty:
            print(f'Face name result not received for visualization after {self.timeout}')
            index, faceName, certainty = None, None, None
        except Exception as e:
            print(f'Error : {e}') 
            index, faceName, certainty = None, None, None
    


