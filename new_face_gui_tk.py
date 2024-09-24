# -*- encoding: utf-8 -*-

import time 
import queue

import cv2 as cv
from PIL import Image, ImageTk 
import tkinter as tk
from tkinter import ttk
from unidecode import unidecode  # To convert UTF-8 strings into ASCII. ex: 'françois' --> 'francois'

import file_management as fl

BLACK = (0,0,0)
RED = (0,0,256)
    
# TODO  Convert it into a GUI class    
def createGUI_tk(img, newFaceIdQueue: queue.Queue):
    """  We ask the user to identify the unknow faces in the new image in unknows_new directory"""
    
    title = 'Stranger identification: (UGLY GUI )'   # title of the window
    imgMsg = 'Unidentified !!'                       # Text on the image    
    questionMsg = f"Please, identify this stranger ? (If you can't let it blank.)"
    answerMsg = f"The stranger name is"
    msgFont = 'Arial 11'
    
    # Create the root Tkinter window object
    root = tk.Tk()   
    root.geometry("600x344")   
    root.title(title)
    
    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 10))
    style.configure('TLabel', font=('Helvetica', 12))
    style.configure('TEntry', font=('Helvetica', 10))


    def close_window():
        root.destroy()
    
    def _return_face_name(user_input):
        """ The face name is either the user input or 'stranger_{next_numero}"""
        
        if user_input is not None and len(user_input)>0:
            return unidecode(user_input.lower())  # convert UTF-8 (e.g: accents... ) in ASCII 
        else:         
            # We count how many kinds of 'stranger' directories are already there.     
            countStrangers = len(fl.listStrangers())  
            return f'stranger_{countStrangers}' 


    def handle_user_input(faceImg = img):
        """  This function uses the following variables that are in its scope:
        question_label, entry, entry_frame, message1_label, message2_label 
        """
        user_input = entry.get().strip()  
        print(f"You entered: {user_input}")                
                        
        faceName = _return_face_name(user_input)            
        question_label.pack_forget()  # Remove the question  
        entry_frame.pack_forget()  # Remove the frame containing the entry and button
        entry.unbind("<Return>")   # Unbind the Enter key (cannot trigger an event again)
                                
        msg1 = f"Hi, we call you \'{faceName}\'."
        msg2 = f"In few seconds (?minutes?), we should recognize {faceName}." 
        message1_label.config(text=msg1)                
        message2_label.config(text=msg2)
        
        # We save the face image in the directory named faceName (which is created if needed)        
        print('Still in createGUI_tk()')     
       
        newFaceIdQueue.put((faceName,faceImg))  #           
        
        close_button = tk.Button(root, text="Close this window", command=close_window)
        close_button.pack(pady=5)
        
    def on_enter(event):
        """Pressing 'Enter' has the same effect than pressing the button
        
        Rem:  The event variable is automatically passed by Tkinter 
                when the <Return> key event occurs. 
        """
        handle_user_input()    
    
    def prepareImgTk(img):
        """  Prepares the face image to be shown in the TK identification GUI. 
        Does not modify the original image, only a local copy  
        """
        img = img.copy()
        # Put some text on a copy of the image:
        cv.putText(img, imgMsg, (2,30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            
        # Convert the image into the Tkinter format 
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)    # Convert to RGB
        img_pil = Image.fromarray(img)              # convert to pil format
        img_tk = ImageTk.PhotoImage(img_pil)        # Convert to Tkinter format
        return img_tk
    
       
    # Create a label and display the image
    img_tk = prepareImgTk(img)
    label = ttk.Label(root, image=img_tk)  #label = tk.Label(root, image=img_tk)
    label.pack()

    # Create a label for the question
    question_label = ttk.Label(root, text=questionMsg, font = msgFont)
    question_label.pack( pady=5)

    # Create a frame for the entry widget and the button
    entry_frame = ttk.Frame(root)
    entry_frame.pack()

    # Create an entry widget to capture user input
    entry = ttk.Entry(entry_frame)
    entry.pack(side=tk.LEFT )  

    # Create a submit button
    submit_button = ttk.Button(entry_frame, text="Enter", command=handle_user_input)
    submit_button.pack(side=tk.LEFT, padx=5)

    # Pressing the button change these message labels
    message1_label = ttk.Label(root, text="", font = msgFont)
    message1_label.pack()
    message2_label = ttk.Label(root, text="", font = msgFont)
    message2_label.pack()

    # Bind the Enter key to the handle_user_input function
    entry.bind("<Return>", on_enter)

    # Tkinter event loop
    root.mainloop()



# =================================================================================
#   ALTERNATIVE  GUI  using tkk instead of tk ( likely less ugly)
# =================================================================================

