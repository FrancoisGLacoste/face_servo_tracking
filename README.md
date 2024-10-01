# face_servo_tracking
Face detection, tracking and recognition + servo-control of arduino robotic head. 

Works so -so.
Still issues regarding:

- Performance (serious lag because of image transfer via mp.queue)
It is being addressed by using mp.shared_memory with mp.queue for image metadata.

- Need to send the recognition result back to imgDisplay where the visualization takes place.
A possibility is to use mp.Queue, another is to use mp.Manager() .

- Everything concerning the GUI where we ask the user to identify unrecognized faces:
Here we use tkinter ( ugly and minimalist...)
It would be better to run a TCP/IP server with Tornado ( for instance). A browser-based GUI would allow the GUI client to run on any machine on the local network, with more flexibility to improve the GUI esthetics.

- Visualization has been partially refactored, but no difference in the display itself:  
still cosmetically non-okay ( ugly and inconsistent choices of colors and fonts....).



