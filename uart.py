# -*- encoding: utf-8 -*-

import serial # pip install pyserial 


class UART:
    
    def __init__(self):
        """ initialize the serial port
        
        Sending 2 int16 values  requires 32 bits. 
        When detection time is around 100 ms  : 32 bits/ 100ms = 320 bits/s   
        If we could have detection time around 50 ms : we would need 640 bits/s, 

        If instead, we send 2 int32 values, it  requires 64 bits, max rate is 640 bit/50 ms 
        i.e At worst, we need BAUD > 1280 
        
        To check the current baud rate of the connected device: 
        stackoverflow.com/questions/38937903/pyserial-how-to-check-the-current-baud-rate
        """
        BAUD = 115200 #9600 # faster is better,  BAUD > 1280 cannot be enough.
        # Sending with higher Baud, should allow the program to spend less time do send data 
        # (and to have less delay between camera state and the servo command)
        port = '/dev/ttyUSB0'  # In Linux,  ls /dev/ttyUSB*
        
        self.ser = self._createSerial(port, BAUD)

    def _createSerial(self,port, BAUD):    
        try: 
            ser = serial.Serial(port, BAUD)
            ser.timeout = 0.2 # [200 ms] 
            return ser
        except Exception as e:#SerialException:
            print("Error with serial port." )
            print(e)   
            return None
        
    def sendData(ser, data):
        """_summary_

        Args:
            ser (_type_): _description_
            data: (np.int16, np.int16) : coordinates to track in the visual field
        """
        
        
        '''def testBaudrateSuitable(): # too NAIVE 
            dataSize = 8 * sys.getsizeof(data)   # [bits]
            maxDataSize = ser.baudrate * tm.getTimeSec()
            assert dataSize < maxDataSize
        '''  
        try:
            ser.write(data)
            return True    
        except Exception as e:
            print(e)
            return False

    