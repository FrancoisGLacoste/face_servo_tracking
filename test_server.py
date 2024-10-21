# -*- encoding: utf-8 -*-

import os
import asyncio
import threading

import tornado.ioloop
import tornado.web
import tornado.httpserver 
import aiofiles

from image_display_v3 import ImageDisplay
from img_transfer import ImgTransfer        

from file_management import DATAPATH  
        

# =======================================================================
# Tornado App : attempt version 2  ( I erased version 1 though...)
#
# ========================================================================    
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
        
        # That indicate we reveive a stream of data (like images) 
        # that should be displayed continuously (like in a video stream). 
        # The boundary parameter is used to separate different parts of the response.
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        try:
            print('Tornado gets in the VideoStreamHandler_v2 get method ***and starts the while loop***')
            imgFilename = os.path.join(DATAPATH, 'audrey', 'audrey158_236.jpg')
            n=0
     
            async for frame in test_generate_img_aiofiles(imgFilename):   
                # We do as if we were fetching different files ( but it is the same one) 
                print(f'The type of the {n}th succesfully retrieved file is {type(frame)}') # class async generator !!
                if n==4: 
                    return  
                """ Rem: We must use 'async for' to iterate over an async generator. 
                ( And not 'await'  )
                """
                if frame is None:
                    break
                
                self.write(b"--frame\r\n")
                self.write(b"Content-Type: image/jpeg\r\n")
                self.write(b"\r\n")
                self.write(frame)
                self.write(b"\r\n")
                
                # Flushes the current output buffer to the network.
                await self.flush() # Ensure the frame is sent. 
                
        except Exception as e:
            # TODO: ERROR: object async_generator can't be used in 'await' expression
            print(f"Error during streaming:{e}")     

def testServer2_main():
    imgTransfert = ImgTransfer()
    server = Server_v2(imgTransfert)
    server.startIOLoop()

# =======================================================================================
#
# Tornado Server for video streaming : version 3
# Like version 2 , but with a list of different pictures from the image directory 
#
# =======================================================================================
class Server_v3:  
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

      
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Here is my wonderfull minimalist GUI. test_server.Server_v3")

class VideoStreamHandler_v3(tornado.web.RequestHandler):
    def initialize(self, imgDisplay):
        super().initialize() 
        self.imgDisplay = imgDisplay 
               
    async def get(self):
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        try:
            async for imgFilename in self.imgDisplay.getFilenames():  
                img = await self.readJpgFile(imgFilename)  
                if img is None:
                    break
                
                self.write(b"--frame\r\n")
                self.write(b"Content-Type: image/jpeg\r\n")
                self.write(b"\r\n")
                self.write(img)  # Send the JPEG bytes
                self.write(b"\r\n")
              
                await self.flush()  # Ensure the frame is sent.
                
        except Exception as e:
            print(f"Error during streaming: {e}")

    async def readJpgFile(self, imgFilename):
        async with aiofiles.open(imgFilename, mode='rb') as f:
            img = await f.read()  # This returns the JPEG bytes
        return img
   

def testServer3_main():
    imgTransfert = ImgTransfer()
    server = Server_v3(imgTransfert)
    server.startIOLoop()
    
# ==========================================================================
#   Minimal Tornado server reading an async generator and displaying the type of the file
#   No video 
# ==========================================================================
class Test_Server_noVideo:  
    def __init__(self, port : int = 8888):
        """ 

        """
        self.port = port
        self.app = self.makeTornadoApp( )
         
    def makeTornadoApp(self):
        """
        """
        return tornado.web.Application([
            (r"/", MainHandler),
       ])

    def startIOLoop(self):
        httpServer = tornado.httpserver.HTTPServer(self.app)
        httpServer.listen(self.port )
        print('tornado ioloop is starting now:')
        tornado.ioloop.IOLoop.current().start()
        print("This line will never be printed: once the ioloop is started, it loops 'forever'.")
  
class MainHandler(tornado.web.RequestHandler):
    
    #def get(self):
    #    self.write("Here is my wonderfull minimalist GUI: like the famous white rabbit in the snowstorm.")
    
    async def get(self):
        print('We are in the MainHandler "get" ' )
        imgFilename = os.path.join(DATAPATH, 'audrey', 'audrey158_236.jpg')
        n=0
        self.set_header("Content-Type", "text/html")  
        async for img in test_generate_img_aiofiles(imgFilename):   
            # We do as if we were fetching different files ( but it the same one) 
            print(f'The type of the {n}th succesfully retrieved file is {type(img)}') # class async generator !!
            if n==4: 
                return
            # Does not work but not important for what I test here (async for... not the page itself)
            
            htmlCode = f""" <html><body><h1>
                                Hello, Here is my GUI: test_min_server!. 
                                The type of the {n}th succesfully retrieved file is {type(img)}
                            </h1></body></html>
                        """
            self.write(htmlCode)
            
            n +=1 
             
def testServer_noVideo_main():
    server = Test_Server_noVideo()
    server.startIOLoop()

# ===========================================================================
# Minimal Server:  Threaded (working) 
#       vs non-threaded (crashes in the async generator: it is expected according to doc)
# ===========================================================================
def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ]) 

def start_loop(app, port):
    #httpServer = tornado.httpserver.HTTPServer(app)
    #httpServer.listen(port )
    app.listen(port)
    print('tornado ioloop is starting now:')
    tornado.ioloop.IOLoop.current().start()
    print("This line will never be printed: once the ioloop is started, it loops 'forever'.")
   
def minimalThreadedServer_main():
    """   To run what Test_Server_noVideo class is supposed to do, but simpler.
    No class definition for the server
    No httpserver
    
    Same problem when trying to read file with aiofiles
    ( Error: cannot schedule new futures after interpreter shutdown)
    """
    app = make_app()
    port =8888
    serverThread = threading.Thread(target=start_loop, args=(app,port))
    serverThread.start()

def minimalServer_main():
    """   To run what Test_Server_noVideo class is supposed to do, but simpler:
      No class definition for the server.
      No httpserver
      No threading.  
    """
    app = make_app()
    port =8888
    #serverThread = threading.Thread(target=start_loop, args=(app,port))
    #serverThread.start()
    start_loop(app,port)
                

# ==========================================================================
#  async for img in test_generate_img_aiofiles(imgFilename) 
# works fine when called in  asyncio.run(test_aiofiles_main()) 
#  ( with no Tornado server)
# ==========================================================================
async def test_read_aiofile(imgFilename: str):
    """Async read an img file using aiofiles 
        return :   the image content"""

    if not os.path.exists(imgFilename):
        print(f"File does not exist: {imgFilename}")
        return None 
    try:
        async with aiofiles.open(imgFilename, mode='rb') as f:
            img = await f.read()
            print(f"File {imgFilename} read successfully: type: {type(img)}") # type: <class 'bytes'>
            return img
    except Exception as e:
        print(f"Error opening file {imgFilename}: {e}")

async def test_generate_img_aiofiles(imgFilename):
    """  Same as test_generateFrame  
        but open only one file: imgFilename
    """
    try:
        img = await test_read_aiofile(imgFilename)
        print(f"In test_aiofiles_main: Frame retrieved successfully type: {type(img)}") # type: <class 'bytes'>
        
        while True:
            await asyncio.sleep(1)
            yield img 
            
    except Exception as e : 
        print(e)            
   
async def test_aiofiles_main():
    """   to test test_aiofiles_generator"""
    
    imgFilename = os.path.join(DATAPATH, 'audrey', 'audrey158_236.jpg')
    try:
        # Test 1: just reading file with aiofiles
        img0 = await test_read_aiofile(imgFilename)
        print(f"In test_aiofiles_main: img retrieved successfully type: {type(img0)}") # type: <class 'bytes'>
        
        # Test 2: reading files from an async generator:
        n=0
        async for img in test_generate_img_aiofiles(imgFilename):
            print(f'We got the {n}th img from the async generator : type: {type(img)}')
            n+=1
            if n> 5:
                break
            
    except Exception as e : 
        print(e)
        
# ===================================================================================================                        
if __name__=='__main__':
    
    # Works:
    #asyncio.run(test_aiofiles_main())
    
    '''    
    # ERROR: cannot schedule new futures after interpreter shutdown
    # When executing the line: 
    #                 async with aiofiles.open(imgFilename, mode='rb') as f
    #           in test_read_aiofile() which is in : test_generate_img_aiofiles()
    # But works fine if the 'get' only writes a lame line... 
    server_thread = Test_Server_noVideo().startInThread()
    
    # Exactement le meme probleme ici (n avec ou sans  httpserver(app))
    minimalThreadedServer_main()
    '''    
    '''# Now it works: WITHOUT threading
    minimalServer_main()
    '''
    '''
    # No Video , like minimalServer_main, but in a class
    testServer_noVideo_main()
    '''
    testServer2_main()
    
