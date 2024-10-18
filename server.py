# -*- encoding: utf-8 -*-

import threading
#import asyncio

import tornado.ioloop
import tornado.web
import tornado.httpserver 
from tornado.websocket import WebSocketHandler

from image_display_v3 import ImageDisplay
from img_transfer import ImgTransfer        
        
# =========  Tornado App ============================
    
class Server:   
    
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
            (r"/video", VideoStreamHandler, dict(imgTransfer = imgTransfer)),
            (r"/results",FacesInfosWebSocket ),
            (r"/file", FileHandler),
        ])

    def startIOLoop(self):
        httpServer = tornado.httpserver.HTTPServer(self.app)
        httpServer.listen(self.port )
        print('tornado ioloop is running')
        tornado.ioloop.IOLoop.current().start()
        print("This line will never be printed: once the ioloop is started, it loops 'forever'.")

    def startInThread(self):
        serverThread = threading.Thread(target=self.startIOLoop)
        serverThread.start()
        return serverThread
  
        
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Here is my wonderfull minimalist GUI: like the famous white rabbit in the snowstorm.")



class VideoStreamHandler(tornado.web.RequestHandler):
    
    """
    VideoStreamHandler is a custom new class that inherits from RequestHandler.
    
    In Tornado, the initialize method is specifically designed for setting up any additional  
    attributes or performing setup tasks when a request handler is instantiated
    """
    def initialize(self, imgTransfer: ImgTransfer):
        super(VideoStreamHandler, self).initialize() 
        self.imgDisplay = ImageDisplay(imgTransfer)
               
    async def get(self):
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        try:
            while True:
                frame = await self.generateFrame()  # Fetch a single frame (in jpg)
                if frame is None:
                    break
                self.write(b"--frame\r\n")
                self.write(b"Content-Type: image/jpeg\r\n")
                self.write(b"\r\n")
                self.write(frame)
                self.write(b"\r\n")
                await self.flush() # Ensure the frame is sent. 
        except Exception as e:
            print(f"Error during streaming:{e}")     
            raise # Topropage the error ???
        
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
                
# =====================================================

class FacesInfosWebSocket(WebSocketHandler):
    
    def open(self):
        " Is executed at the opening of the websocket."
        print('websocket opened')
        
        
    def on_message(self, message):
        "Receive and display incoming message."
        self.write_message(f'Received: {message}')
    
    def on_close(self):
        " Executed when websocket is closed."
        print('Perhaps we should do some cleaning here ??')
        print('Websocket is now closed.')
        
              
        
        
#   =====================================================
import mimetypes

class FileHandler(tornado.web.RequestHandler):
    def get(self, filename):
        # Determine the MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            self.set_header("Content-Type", mime_type)
        else:
            self.set_header("Content-Type", "application/octet-stream")  # Fallback

        # Serve the file (not shown here)
        self.write("Serving file: " + filename)  # Replace with actual file serving logic

# ===========================================================
if __name__=='__main__':
    imgTransfert = ImgTransfer()
    server = Server(imgTransfert)
    server.startInThread()