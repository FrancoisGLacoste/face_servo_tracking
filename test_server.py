# -*- encoding: utf-8 -*-

import threading
import os

import tornado.ioloop
import tornado.web
import tornado.httpserver 
#from async_files import FileIO
import aiofiles
import asyncio

from image_display_v3 import ImageDisplay
from img_transfer import ImgTransfer        

from file_management import DATAPATH  #, readImgFiles
        
# =========  Tornado App : attempt version 1 ============================
    
class Server_v1:  
    def __init__(self, imgTransfer: ImgTransfer, port : int = 8888):
        self.port = port
        self.app = self.makeTornadoApp(imgTransfer )
         
    def makeTornadoApp(self, imgTransfer: ImgTransfer ):
        """
        imgDisplay is initialized in the handler using imgTransfer, 
        but imgTransfer itself is not retained as an attribute.
        """
        return tornado.web.Application([
            (r"/", MainHandler),
            (r"/video", VideoStreamHandler_v1, dict(imgTransfer = imgTransfer)),
        ])

    def startIOLoop(self):
        httpServer = tornado.httpserver.HTTPServer(self.app)
        httpServer.listen(self.port )
        print('tornado ioloop is starting now:')
        tornado.ioloop.IOLoop.current().start()
        print("This line will never be printed: once the ioloop is started, it loops 'forever'.")

    def startInThread(self):
        serverThread = threading.Thread(target=self.startIOLoop)
        serverThread.start()
        return serverThread
  


class VideoStreamHandler_v1(tornado.web.RequestHandler):
    
    """
    ATTENTION: ICI VideoStreamHandler_v1 initialization only takes place when we open 
    localhost:8888\video page in the browser.
    
    VideoStreamHandler is a custom new class that inherits from RequestHandler.
    
    In Tornado, the initialize method is specifically designed for setting up any additional  
    attributes or performing setup tasks when a request handler is instantiated
    """
    def initialize(self, imgTransfer: ImgTransfer):
        super().initialize() 
        self.imgDisplay = ImageDisplay(imgTransfer)
        print('Here we are in VideoStreamHandler. *****')
               
    async def get(self):
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        try:
            print('Tornado gets in the Tornado get method ***and starts the while loop***')
            while True:
                '''
                frame = await self.generateFrame()  # Fetch a single frame (in jpg)
                if frame is None:
                    break
                self.write(b"--frame\r\n")
                self.write(b"Content-Type: image/jpeg\r\n")
                self.write(b"\r\n")
                self.write(frame)
                self.write(b"\r\n")
                await self.flush() # Ensure the frame is sent. 
                '''
        except Exception as e:
            # TODO: ERROR: object async_generator can't be used in 'await' expression
            print(f"Error during streaming:{e}")     

    async def generateFrame(self ):
        """   Generator that returns (yields) frames in JPEG format"""
        while True:
            try:
                frame = await self.imgDisplay.prepareFrame_sync() 
                if frame is None:
                    break  # end of stream 
                
                yield frame
            except Exception as e:
                print(e)
                


# =========  Tornado App : attempt version 2 ============================
    
class Server_v2:  
    def __init__(self, imgTransfer: ImgTransfer, port : int = 8888):
        """ 
        imgDisplay is initialized in the handler using imgTransfer, 
        but imgTransfer itself is not retained as an attribute.
        """

        self.port = port
        imgDisplay = ImageDisplay(imgTransfer)  
        self.app = self.makeTornadoApp( imgDisplay)
         
    def makeTornadoApp(self,imgDisplay: ImageDisplay):
        """
        """
        return tornado.web.Application([
            (r"/", MainHandler),
            (r"/video", VideoStreamHandler_v2, dict(imgDisplay = imgDisplay)),
        ])

    def startIOLoop(self):
        httpServer = tornado.httpserver.HTTPServer(self.app)
        httpServer.listen(self.port )
        print('tornado ioloop is starting now:')
        tornado.ioloop.IOLoop.current().start()
        print("This line will never be printed: once the ioloop is started, it loops 'forever'.")

    def startInThread(self):
        serverThread = threading.Thread(target=self.startIOLoop)
        serverThread.start()
        return serverThread
  
        
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Here is my wonderfull minimalist GUI. test_server.Server_v2")


class VideoStreamHandler_v2(tornado.web.RequestHandler):
    
    """
  
    VideoStreamHandler is a custom new class that inherits from RequestHandler.
    
    In Tornado, the initialize method is specifically designed for setting up any additional  
    attributes or performing setup tasks when a request handler is instantiated
    """
    
    def initialize(self, imgDisplay: ImageDisplay):
        super().initialize() 
        self.imgDisplay = imgDisplay 
        print('Here we are in VideoStreamHandler_v2. *****')
               
    async def get(self):
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        try:
            print('Tornado gets in the VideoStreamHandler_v2 get method ***and starts the while loop***')
            """while True:
                frame = await self.generateFrame()  # Fetch a single frame (in jpg)
                # In this example, no frame are available to fetch: 
                # It must handle this case    
                # And in the case frames are arriving but this page is not ready: 
                # we should empty the queue when the content is not 'consumed'(received and used)
            """
            async for frame in self.test_generateFrame():  
                # Fetch a single frame (in jpg)
                """We must use 'async for' to iterate over an async generator. 
                ( And not 'await'  )
                """
                if frame is None:
                    break
                self.write(b"--frame\r\n")
                self.write(b"Content-Type: image/jpeg\r\n")
                self.write(b"\r\n")
                self.write(frame)
                self.write(b"\r\n")
                await self.flush() # Ensure the frame is sent. 
                
        except Exception as e:
            # TODO: ERROR: object async_generator can't be used in 'await' expression
            print(f"Error during streaming:{e}")     

    
    
    async def test_generateFrame(self):
        """ Asynchronous generator that just yields the same single JPG image from a file. 
        We use aiofiles because we must access to the file in an async way,   """    

        try:
            '''
            imgFilename = os.path.join(DATAPATH,'audrey','audrey158_236.jpg')    
            async with aiofiles.open(imgFilename, mode='rb') as f:
                frame = await f.read()  # Read the entire file asynchronously
                while True:
                    await asyncio.sleep(1)  
                    yield frame  # Yield the same frame 
            '''
            frame0 = await test_read_aiofile()  
            while True:
                await asyncio.sleep(1)
                frame = frame0.copy()  
                yield frame  # Yield the same frame   
                   
        except Exception as e: 
            print(e)    
            
# ===========================================================

async def test_read_aiofile():

    imgFilename = os.path.join(DATAPATH, 'audrey', 'audrey158_236.jpg')

    if not os.path.exists(imgFilename):
        print(f"File does not exist: {imgFilename}")
        return None 

    try:
        async with aiofiles.open(imgFilename, mode='rb') as f:
            frame = await f.read()
            print(f"File {imgFilename} read successfully: type: {type(frame)}") # type: <class 'bytes'>
            return frame
    except Exception as e:
        print(f"Error opening file {imgFilename}: {e}")
        return None

async def test_aiofiles_main0():
    
    try:
        frame = await test_read_aiofile()
        print(f"In test_aiofiles_main: Frame retrieved successfully type: {type(frame)}") # type: <class 'bytes'>
        
    except Exception as e : 
        print(e)

async def test_aiofiles_generator():
    
    try:
        img = await test_read_aiofile()
        print(f"In test_aiofiles_main: Frame retrieved successfully type: {type(img)}") # type: <class 'bytes'>
        
        while True:
            await asyncio.sleep(1)
            yield img 
            
    except Exception as e : 
        print(e)            
   
async def test_aiofiles_main1():
    """   to test test_aiofiles_generator"""
    try:
        img0 = await test_read_aiofile()
        print(f"In test_aiofiles_main: img retrieved successfully type: {type(img0)}") # type: <class 'bytes'>
        
        n=0
        async for img in test_aiofiles_generator():
            print(f'We got the {n}th img from the async generator : type: {type(img)}')
            n+=1
    except Exception as e : 
        print(e)
                        
if __name__=='__main__':
    '''imgTransfert = ImgTransfer()
    #server = Server_v1(imgTransfert)
    server = Server_v2(imgTransfert)
    serverThread = server.startInThread()
    '''
    
    asyncio.run(test_aiofiles_main1())
    