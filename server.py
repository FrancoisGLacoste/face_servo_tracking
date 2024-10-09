# -*- encoding: utf-8 -*-



import tornado.ioloop
import tornado.web
from tornado.web import RequestHandler, VideoStreamHandler 

from image_display_v3 import ImageDisplay
from img_transfer import ImgTransfer        
        
# =========  Tornado App ============================
    
def make_app(imgTransfer: ImgTransfer):
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/video", MyVideoStreamHandler, dict(imgTransfer = imgTransfer)),
        """imgDisplay is initialized in the handler using imgTransfer, 
        but imgTransfer itself is not retained as an attribute."""
    ])


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")



class MyVideoStreamHandler(VideoStreamHandler):
    #VideoStreamHandler(tornado.web.RequestHandler):
    """
    In Tornado, the initialize method is specifically designed for setting up any additional  
    attributes or performing setup tasks when a request handler is instantiated
    """
    def initialize(self, imgTransfer: ImgTransfer):
        super(MyVideoStreamHandler, self).initialize() 
        self.imgDisplay = ImageDisplay(imgTransfer)
               
    async def get(self):
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        while True:
            frame = await self.get_frame(self.imgDisplay)  # Fetch a single frame
            if frame is None:
                break
            self.write(b"--frame\r\n")
            self.write(b"Content-Type: image/jpeg\r\n")
            self.write(b"\r\n")
            self.write(frame)
            self.write(b"\r\n")
            await self.flush()

    async def get_frame(self ):
        while True:
            try:
                frame = await self.imgDisplay.prepareFrame() 
                if frame is None:
                    break  # end of stream 
                
                # encode in jpeg:
                # HOW ??
                
                yield frame
            except Exception as e:
                print(e)
                
  
        
        
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
if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()