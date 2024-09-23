# -*- encoding: utf-8 -*-


import os
from os.path import join, dirname, splitext, isdir
from os import listdir

import numpy as np 
import cv2 as cv
from cv2 import imread
from PIL import Image
import imghdr
import json 

from trajectory import Trajectory


# Finding the path of the base directory i.e path were this file is placed
BASE_DIR = dirname(os.path.abspath(__file__))
#BASE_DIR2 = '/home/moi1/Documents/dev_py/vision/PROJET_Face-Tracking-camera/'

DATAPATH = join(BASE_DIR,'faces_data')
   
YUNET_DETECTION_PATH = join(BASE_DIR, 'face_detection_yunet_2023mar.onnx')   
VIT_TRACKING_PATH = join(BASE_DIR, 'object_tracking_vittrack_2023sep.onnx')   
# =======================================================================================
#        File management for trajectories                                               #     
# =======================================================================================       

class NpEncoder(json.JSONEncoder):
    """ 
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable 
     
        data_json= json.dumps(data, cls=NpEncoder)
        json.dumps(data_json, file)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
   
def saveTraject(traject:Trajectory, mode):
    n=1
    # If faceCenterTraject is long enough, we save it in JSON to represent
    # the non-filtered signal behavior in the case of each mode 
    # (detection mode and tracking mode)  
    traject_filename = join(BASE_DIR, f'faceCenterTraject_{n}_{mode}.json')

    if len(traject.observations) > 20:
        with open(traject_filename, 'w') as f:
            data = {'coord': traject.observations, 
                    'time':traject.observationsTime}
            json.dump(data, f, cls=NpEncoder)
 
  
def openTraject(file):
    """We open and plot the trajectory To visualize how we should filter it."""
    
    file = join(BASE_DIR, file)
    
    with open(file) as f:
        data = json.load(f) # str
    
    coordTraject  = data["coord"] #format: [(x,y), ...]
    x = [c[0] for c in coordTraject]
    y = [c[1] for c in coordTraject]
    t_ = data["time"] 
    #print(type(t_))    # list
    #print(type(t_[0])) # float
    t = [ (t - t_[0]) for t in t_ ]
    return x,y,t 
   
   
# =======================================================================================
#        File management for images                                                     #     
# =======================================================================================       
def isValidJPG(f):
    """DeprecationWarning: 'imghdr' is deprecated 
    and slated for removal in Python 3.13"""
    return(imghdr.what(f)=='jpeg')

def saveVideo(video):
    # TODO 
    # Initialize video writer
    fps = video.get(cv.CAP_PROP_FPS)
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    frame_size = (width,height)
    output_video = cv.VideoWriter('output.mp4', 
                                    cv.VideoWriter_fourcc(*'mp4v'), 
                                    fps, 
                                    frame_size) 
    NotImplemented
    
    return output_video
    
def isNameDir(name_dir):
    return isdir(join(DATAPATH,name_dir))
  
def createNameDir(name_dir):
    """  Create a directory named {name_dir},
        that is used to contain the face images called {name_dir}_{numero}.
    """
    try: 
        os.mkdir(join(DATAPATH,name_dir))
    except FileExistsError as e:
        print(e)        
           
def listFaceNames():
    """
    Returns  the list of all face_names that are not stranger.

    Assuming:
         Filenames in in DATAPATH either
                correspond to our face names ('{face_name}'),
                or  be 'stranger', 'stranger_new' or '{face_name}_new'"""
    return [f for f in listdir(DATAPATH) 
                  if   isNameDir(f)
                  and '_new' not in f 
                  and  'stranger' not in f 
            ] 
    
def listStrangers():
    return [f for f in listdir(DATAPATH) 
                  if   isNameDir(f)
                  and  'stranger' in f 
            ] 
 
                    
def readFaceImgs_v1(filename = 'strangers_new'):
    """ returns a list of all images of name <name> """
    name_dir = join(DATAPATH, filename.lower())
    imgs = list()
    for f in listdir(name_dir):   
        if splitext(f)[1] in ['.jpg', '.jpeg']:   
            try:     
                print(cv.imread_defines)
                # "IMREAD_LOAD_GDAL" and "IMREAD_LOAD_IMAGE_REDUCER"
                im = imread(join(name_dir,f)) 
                print(type(im))
                print(im.shape)
                if im is None: 
                    continue
                imgs.append(im)
            except Exception as e:
                print(e)
                continue
    return imgs
    
def yieldFaceImgs(face_name):
    """  Similar to readFaceImgs, but as a generator rather than a list"""
    # for each face_name in a face_name directory:  yield all image file 
    name_dir = join(DATAPATH, face_name.lower())
    imgs = list()
    for f in listdir(name_dir):   
        if splitext(f)[1] in ['.jpg', '.jpeg']:   #isValidJPG(f): 
            try:     
                im = imread(join(name_dir,f)) 
                if im is None: 
                    continue
                yield im
            except Exception as e:  
                print(e)
                continue    
    
    
def readImgFiles(dirname = 'strangers_new', number=-1):
    """  Reads and returns at most {number} images in directory {filemame} 
    
    Default number =-1 means we loop over all files in {name_dir}
    
    returns:  list of (id,im) , for id, f in enumerate(listdir(name_dir))
    """
    
    name_dir = join(DATAPATH, dirname)
    imgs = list()
    #correctFiles = list()
    for id, f in enumerate(sorted(listdir(name_dir))):
        if id==number: 
            break  
        #print(f) # relative path, not absolute path
        try:
            filepath = join(name_dir,f)
            if not isValidJPG(filepath) : 
                print(f'{id}: The file {f} is not a valid jpeg !!??')
                continue
            im = imread(filepath)
            if im is None:
                print(f'File {f} has no image despite being a valid jpeg !')
                continue
            print(f'{id}: File {f} has been read.')
        except Exception as e:
            print(f"{id}: Error processing file {f}: {e}")
            continue
        
        imgs.append((id,im))
        #correctFiles.append(f)
        
    #invalidFiles = [f for f in listdir(name_dir) if f not in correctFiles]  
    #print(f'The invalid files are: {invalidFiles}')
    #print(len(invalidFiles)) 
    if len(imgs) != len(listdir(name_dir)):
        print('Warning: Some images have not been saved.. ')    
    return imgs
     
        
    
def giveNewFilepath(face_name, id=None):
    # Assuming each directory is named after the name of the face
    # and contains only files that are faces images '<face_name>_<number>.jpg
    
    # When id is not None: Saved in .../{face_name}/{face_name}_{id}_{count+1}
    
    face_name = face_name.lower()
    name_dir = join(DATAPATH, face_name)   
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)
    if id is not None: 
        face_name += f'{id}'         
    filename = f'{face_name}_{count(name_dir)+1}.jpg'
               
    return join(name_dir, filename)    
     
     
def cropBoxes(img, boxes, inGray=False):  
    """ 
    Arg: 
        img:   np.ndarray         a single image
        boxes: np.ndarray([:,:4]) or np.ndarray([:,:15]) : array of boxes, i.e. coords (x,y,w,h), e.g of the faces.
    Returns:    list of  images contained by the boxes, (gray if asked) 
    
    
    REM: SFace model has a method alignCrop(image, face_box)
    """
    #print(boxes[0]) # valid both for lists and arrays
    if boxes is None or len(boxes.squeeze())==0: 
        print('No box to crop from the image')
        return []
    if inGray : img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
    
    return [img[y:y+h,x:x+w] for (x,y,w,h) in boxes[:,:4].astype(int)]
      
    
def count(name_dir):
    return len(listdir(name_dir) ) 

def saveFaceImg(face_name,face_img, file_id=None):
    """Save  the face_image in file path = 
                .../faces_data/{face_name}/{face_name}_{numero}.jpg
    Args:
        face_name (string): face name, image label
        face_img (np.array): face image ( not the box coord) 
    """
    # Saved in .../{face_name}/{face_name}_{id}_{count+1}
    filepath = giveNewFilepath(face_name, file_id) 
    jpgformat = [cv.IMWRITE_JPEG_QUALITY, 90]
    try:
        cv.imwrite(filepath, np.ascontiguousarray(face_img), jpgformat)
    except Exception as e:
        print(e)
        print(f'Saving the face image in {filepath} from image file #{file_id} has failed !! :( ') 
            
def save_face_imgs(face_names,face_imgs,file_ids, name2save =None):
    """ 
    face_names  list    face name
    face_imgs   list    face image corresponding to face name
    file_ids     list   file id corresponding to the image and name (all aligned in list)
    name2save: List of the face names for which we save the face image
               can include 'stranger', person names
               When None, it means all names are saved 
    """
    if name2save is None: name2save = face_names # a list of face names
   
    notSaved = list()
    ids_names_imgs = zip(file_ids,face_names,face_imgs)
    for n,(file_id, face_name, face_img) in enumerate(ids_names_imgs):
        #print(type(face_img))
        #print(face_name)
        if face_name in name2save and face_img is not None:                 
            try: 
                saveFaceImg(face_name,face_img,file_id) 
                print(f'{n}: a face from file #{file_id} of {face_name} has been saved.')      
                
            except SyntaxError as e : 
                print(e)  # ValueError: ndarray is not C-contiguous
                print(f'{n}: a face from file #{file_id} of {face_name} has NOT been saved  :( ') 
                notSaved.append((id,face_name,face_img))
                continue
    if len(notSaved)==0: 
        print('No error have been caught during saving')                    
    return notSaved


def saveNewFaceImg(face_name, faceImg):
    if not isNameDir(face_name):
        createNameDir(face_name)
        print(f'We just created a directory named \'{face_name}\' ')

    saveFaceImg(face_name,faceImg)
    print(f'We just saved the new face image in the directory named \'{face_name}\' ')

   
def test_readImgs_cv():  
    name_dir = join(DATAPATH, 'audrey_new')
    imgs = list()
    correctFiles = list()
    for f in listdir(name_dir):  
        print(f)     # relative path, not absolute path
        try:
            if not isValidJPG(join(name_dir,f)) : 
                print(f'Looks like {f} is not a valid jpeg !!??')
            im = imread(join(name_dir,f))
            if im is None:
                print(f'file {f} has no image despite being a valid jpeg ?!')
                continue
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            continue
        
        imgs.append(im)
        correctFiles.append(f)
        
    invalidFiles = [f for f in listdir(name_dir) if f not in correctFiles]  
    print(f'The invalid files are: {invalidFiles}')
    print(len(invalidFiles)) 
    ###  REM:  the message "Invalid SOS parameters for sequential JPEG" is NOT an error.
    ### All the jpg files are valid and properly processed. 
    assert len( imgs) == len(correctFiles)
    assert len(invalidFiles) == 0
    print('All good !') 

                    
if __name__ == '__main__':
    
    imgs=test_readImgs_cv()  #                 