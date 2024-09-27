# face_servo_tracking
Face detection, tracking and recognition + servo-control of arduino robotic head



---------------------------------------------------------------------------------------------------------------
Warning: the main branch runs : the kalman filtering is however not applied to all trajectories. (corrected in Development branch)
- There is still a lot of work to do on the criterion that decides if an unrecognized face has to be sent to the GUI for identification.
- The GUI itself has still to be refactored into an object. The GUI looks like a retro-futuristic parody of the 1990s and is so ugly it becomes funny. 

- Serious work is also needed on criterion that classifies faces as unrecognized. ( For now I use a simple threshold on the distance to centroid for each cluster, but it overfits too much. I want to use the Local Outiler Factor method (LOF) )

- All the visualization module is to refactor.
- In the refactoring, I will also create a new class that encapsulates specific attributes and methods of FaceImage. 
- I must find a way to send the recognition task result ( the recognized face name) in the function that displays the image. 

- There is finally a performance issue with the queues and the threads: sometimes, a queue is full too fast and data are lost. Other times, the recognition task freezes while the new face images to process accumulate in the queue.
Is it that I have to find a way to prioritize the face recognition task threads ?
Here some suggestions from the AI assistant: 
https://poe.com/s/4yQH2vztQ58OEqsB5xy9
