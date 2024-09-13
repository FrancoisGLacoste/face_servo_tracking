# -*- encoding: utf-8 -*-

"""
face_recognition.py 
"""
import os 
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


import cv2 as cv
import numpy as np

from PIL import Image, ImageTk #TODO should be in the image file /or class ?
import tkinter as tk

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from unidecode import unidecode  # To convert UTF-8 strings into ASCII. ex: 'françois' --> 'francois'
import file as fl
import util as u

BLACK = (0,0,0)
RED = (0,0,256)
"""  ====================== SFACE ======================================= 
SFace : [Zhonyu2021] a state-of-the-art algorithm for face recognition
Ref:
https://github.com/zhongyy/SFace/blob/main/SFace_torch/train_SFace_torch.py
https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py#L99

======================================================================="""


class FaceRecognition:
    
    def __init__(self,isActive = True, faceNames2check= None):
        self.isActive = isActive       
        self.faceNames = self.setFaceNames(faceNames2check) # face names we want to recognize
         
        self.inputQueue = mp.Queue()       # Queue for moving face images to the face recognition task
        self.resultQueue = asyncio.Queue() # Queue for capturing the recognition results (i.e. face names)
        self.executor = ThreadPoolExecutor(max_workers=2) # Thread pool for the face recognition task        
        
        self.recognizer = cv.FaceRecognizerSF.create(
                            os.path.join(fl.BASE_DIR,'face_recognition_sface_2021dec.onnx'), 
                            "" )
        
        self.faceEmbeddingsDict =dict()  
        #self.X = None    # (X,y)  : dataset to be loaded, based on faceEmbeddingsDict 
        #self.y = None
        
        self.LogisticRegressionClf = LogisticRegression(random_state=0)
        
        # k-nearest-neighbors classifiers with learnable thresholds 
        self.kNNClfs = dict()  #  Dictionary of kNN classifiers with respect to various metrics
        self.kNNClf_thresholds = dict()
        self.kNNClf_ratioThresholds = dict()
        
        #  n_neighbors: to be determine via cross-validation
        # some possibily useful methods : get_params(), set_params()
        #                                 kneighbors()  : find the k neighbors and their distances  
        #                                 predict(), predict_proba(X)      
        # scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
                    
                    
    def putImgQueue(self, img):
        """ Put the face image in the queue for face recognition. 
        The image will show up in runFaceRecognitionTask()        
        """
        self.inputQueue.put(img)
    
        
    async def runFaceRecognitionTask(self):
        print('we arrive in runFaceRecognitionTask')
        while True:
            # Receive a face image from the face detection loop
            print('We are awaiting for the last face image that has been detected.')
            
            faceImgs = await asyncio.get_event_loop().run_in_executor(self.executor,self.inputQueue.get)
            print('We got a new face image for the face recognition task to process.')
                
            #results =  [await self.recognizeFace(img) for img in faceImgs]
            results=list()
            for img in faceImgs:
                recognizedName, certainty = await self.recognizeFace(img)
                results.append((img,recognizedName, certainty)) 
          
            # Put the result in the result queue: the result is sent to retrieveResults()
            await self.resultQueue.put(results) 

      
    async def retrieveResults(self):
        """ Retrieves results from face recognition of the face images 
            from a video frame.
           The unrecognized face are identified by the user, and saved as {name}_{count+1}.jpeg 
           the new directory {name} 
        """
        while True:
            try:
                print('Awaiting for result in retrieveResults')
                result = await self.resultQueue.get()
                print(f"Result received: {result}")
                self.resultQueue.task_done()
                
                
                for img, name,_ in result:
                    if img is not None:
                        #if name == 'stranger': # i.e. unrecognized 
                        if name in ['francois', 'audrey', 'stranger']:
                            self.createGUI_tkinter(img)

            except asyncio.CancelledError as e:
                print('The task has been canceled, exit the loop')
                break
            except Exception as e:
                # Handle any other exceptions that might occur
                print(f"Error in run() function: {e}")
                # Notify the queue that the task is done
                self.resultQueue.task_done()


    def embeddingsAreSaved(self):
        """ 
        Returns a Boolean value: True when we find a file that contains the SFace features embeddings 
        
        """
        # TODO
        return False
    
    def loadEmbeddings(self):
        # TODO
        NotImplemented
            
    def kNNClfsAreTrained(self):
        # TODO
        return False 
        
    def loadkNNClfs(self):
        #TODO
        NotImplemented  
    
    def setFaceNames(self,faceNames2check =None):
        if faceNames2check is None:
            #Default: list face_names i.e. directory names in DATAPATH, except the ones that end with '_new'
            self.faceNames = fl.listFaceNames() # ex: ['audrey', 'francois', 'victor', ']
        else:
            self.faceNames = faceNames2check 
      
    def prepareFaceRecognition(self):   # TODO
        
        self.setFaceNames(['audrey','francois'])
        
        # Compute or load self.faceEmbeddingsDict[face_name] 
        if not self.embeddingsAreSaved():
            self.computeAllFaceEmbeddings() 
        else: 
            self.loadEmbeddings()
        
        # Compute or load the knn classifier (self.kNNClfs[metric] )   
        if not self.kNNClfsAreTrained():    
            X, y = self.convertDatasetInArrays()
            self.trainKNNClfs(X, y)
        else: 
            self.loadkNNClfs()  
        
        # TODO:Compute or load eventual additional classifiers (logistic regression...)
        print('FaceRecognition preparation has been done.')      
    
    # =========================    Face feature embeddings ==============================    
    def computeFaceEmbeddings(self, face_names):
        """Train SFace algorithm by computing face embeddings.
        
        It prepares a dictionary of the embeddings of face features of each face name
        self.faceEmbeddingsDict = { face_name1:  [img1features, img2features ...], 
                                    face_name2: [...],
                                    ...}
        """
        for face_name in face_names:
            embeddings = []
            for face_img in fl.yieldFaceImgs(face_name):     #np.ndarray
                # REM: ils utilisaient aligncrop() pour obtenir face_img
                embedding = self.recognizer.feature(face_img)
            
                #print(embedding.shape)   # (1,128)
                embeddings.append(embedding)
            
            self.faceEmbeddingsDict[face_name]  = (np.array(embeddings)).reshape(len(embeddings), 128)
            
        
    def computeAllFaceEmbeddings(self):
        """ 'Train' SFace algorithm by computing the face embeddings 
           for all face names that exists in DATAPATH   
        """
        if self.isActive:
            self.computeFaceEmbeddings(self.faceNames)
            print('We have just computed all face embeddings.')
        else:
            print('Face recognition is inactive: no embedding computed.')
            
                    
    def updateFaceEmbeddings(self): # TODO  
        """  Updates face embeddings with the new face images that have just been 
        identified by the users. 
        
        It can be new images with new names (in a new directory of the same name)
        or 
        can be new images associated with known names. 
        """
        # TODO
        filenames = ''
        self.computeFaceEmbeddings(filenames)
    
    def convertDatasetInArrays(self):
        """   
        Face Embeddings is a dictionary of list of image feature embeddings: 
                            {'audrey': [img_embed1, img_embed2, ...], 'francois':[...]}
                            where img_embed1, img_embed2, ...are np.array(1,d) , d=128 dimensions
            
        But sci-kit-learn methods expect the dataset to be in np.arrays
        i.e. 
        X : np.array(n, d)   : data points
        y : np.array(1,n)    : array of label indices ( i.e. each index represents a face name)
        """
        
        for n, faceName in enumerate(self.faceEmbeddingsDict.keys()):
            #embed_list = self.faceEmbeddingsDict[faceName] # list of array(1,128)
            #y = n*len(embed_list)
            
            X_n = np.array(self.faceEmbeddingsDict[faceName])  # array(Nn, 128)
            lenght = X_n.shape[0]
            y_n = (np.array([n]*lenght)).reshape(lenght,)     # array(Nn, )
            
            # Stack the arrays X <--- [ X|X_n ]
            if n == 0:
                X = X_n
                y = y_n 
            else:
                X = np.vstack([X,X_n])
                y = np.hstack([y,y_n])
        return X,y
    
    
    # ================================   A first naive match criterion =========================================    
    def _computel2Scores(self, newFace_features, faceName, l2_similarity_threshold = 1.128):
        l2Scores = list()
        for face_features in self.faceEmbeddingsDict[faceName]:
            l2Scores.append(self.recognizer.match(newFace_features, face_features, 
                                             cv.FaceRecognizerSF_FR_NORM_L2)
                            )
        l2Match = [(l2Score <= l2_similarity_threshold) for l2Score in l2Scores]
        return l2Scores, l2Match
    
    def _computeCosineScores(self, newFace_features, faceName,cosine_similarity_threshold = 0.363):
        cosineScores = list()
        for face_features in self.faceEmbeddingsDict[faceName]:   
            cosineScores.append(self.recognizer.match(newFace_features, face_features, 
                                                 cv.FaceRecognizerSF_FR_COSINE)
                                )
    
        cosineMatch= [(cosScore >= cosine_similarity_threshold) for cosScore in cosineScores]
        return cosineScores, cosineMatch
       
    def isAMatch_positiveFraction(self, faceImg, faceName):
        """  Detects a likely match between faceImg and faceName based on the positive fraction of 
        cosine similarity and l2 similarity.
        
        Args:
            faceImg: np.ndarray : the face image we want to recognize. 

            faceName: string: face name
        
        Returns:  fraction of element in the embeddings[faceName] list that match with respect to 
                  both cosine_similarity and l2_similarity, given the pre-defined thresholds
        
        """
        
        newFace_features = self.recognizer.feature(faceImg)
        cosineScores, cosineMatches= self._computeCosineScores(newFace_features, faceName)
        l2Scores, l2Matches = self._computel2Scores(newFace_features, faceName)
        doesMatch = cosineMatches and l2Matches 
        positiveFraction = len([e for e in doesMatch if e])/len(doesMatch)
        return positiveFraction

    # TODO ??  to delete when we obtain a good classifier that works 
    async def recognizeFace_v1(self, unknownFaceImg):
        """
        Performs face recognition with SFace on a specific face image in arg
        """
        positiveFraction = dict()
        for faceName in self.faceNames:
            positiveFraction[faceName]  = self.isAMatch_positiveFraction(unknownFaceImg, 
                                                                        faceName)
    
        recognizedName = u.argmax_dict(positiveFraction)
        print(f'The stranger face is more likely {recognizedName}')
        
        return (recognizedName, positiveFraction[recognizedName])
 

    # ============ Logistic regression classifier to apply over the SFace features embeddings===========
    # TODO   
    async def recognizeFace_LogisticRegression(self, unknownFaceImg):
        """ https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        """
        newFace_features = self.recognizer.feature(unknownFaceImg)
        
        self.LogisticRegressioncClf.predict_proba(newFace_features[:2, :])
        self.LogisticRegressioncClf.predict(newFace_features[:2, :])
        #self.LogisticRegressioncClf.score(newFace_features, y)
        NotImplemented   # TODO
            

    def train_LogisticRegression(self, X, y):
        self.LogisticRegressioncClf.fit(X, y)
        NotImplemented  # TODO

    # ========   k-nearest-neighbors classifiers to apply over the SFace features embeddings =====
    
        
    async def recognizeFace(self, unknownFaceImg):
        """ 
        Returns tuple (predictedFaceName, predictionCertainty)
        """
        newFace_features = self.recognizer.feature(unknownFaceImg)
               
        # i.e. {metric : (faceName, predict_proba) for metric in [l2,cosine]}  
        predict_proba = { metric:  self.predict_kNNClf(newFace_features, metric ) 
                            for metric in ['l2', 'cosine']
                        } 
    
        # faceName associated with the metric whose predictor has the best result      
        
        faceNames_proba_list = list(predict_proba.values()) # [(faceName0, proba0), (faceName0, proba0)]
        id_list = list(predict_proba.keys())
        print(faceNames_proba_list)
        # faceName such that proba is max in [(faceName_l2, proba_l2), (faceName_cos, proba_cos)]
        faceName, prob_faceName = u.argmax_tupls(faceNames_proba_list )
        
        print(f'in recognizeFace: faceName={faceName} with prob={prob_faceName}')
        return faceName, prob_faceName
        
    def trainKNNClfs(self,X, y):
        """  Trains k-nearest-neighbors classifiers
        
        X: np.array (n, d):   n data points in d-dimensions
        y: list len(y)=n , OR np.array (1,n): data labels   
        """        
        # Determine distance threshold from validation set
        def find_dist_threshold( X_val, y_val, quantile=0.98):
            pipeline = self.kNNClfs[metric]
            distances, _ = pipeline['knn'].kneighbors(X_val) # kneighbors =??
            # Calculate distances for correct classifications
            correct_distances = [
                distances[i, 0] for i in range(len(y_val))
                if pipeline.predict(X_val[i].reshape(1, -1)) == y_val[i]
            ]
            # Use a high quantile to set the threshold
            threshold = np.quantile(correct_distances, quantile)
            return threshold

        def find_probaRatioThreshold(X_val, y_val):
            """Returns the optimal threshold for the ratio between first ans second nearest classes. 
            
            The closer to 1 is the ratio, the lower is the classification signicance. i.e the actual
            classification might be 'stranger'. 
            """
            threshold=1 
            return threshold 
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

        # For both metric (l2 and cosine), find hyperparameters for the "best" kNN model :
        for metric in ['l2', 'cosine']:  
            # Pipeline with data scaling 
            pipeline = Pipeline([ ('scaler', StandardScaler()),
                                    ('knn', KNeighborsClassifier(  weights = 'distance', 
                                                                    metric  = metric))
                                ])    
            
            # Cross-validation with parameter grid search to find the best k= n_neighbors
            # Parameters of pipelines can be set using __ separated parameter names, 
            param_grid = {'knn__n_neighbors': [1,2,3,5,8]}
            
            grid_search = GridSearchCV(pipeline, param_grid, cv=5)
            grid_search.fit(X_train, y_train)   # shape of y : (n_samples,)

            # Choose the Best kNN model (according to the choice of hyperparam k )
            # best_estimator_ :  type = Pipeline, NOT type KNeighborsClassifier
            self.kNNClfs[metric] = grid_search.best_estimator_ 
            
            model = self.kNNClfs[metric]['knn']  # type : KNeighborsClassifier
            print( f'k-nearest-neighbor with k={model.n_neighbors} and metric={metric}')
            
            # Get the threshold from validation data            
            #self.kNNClf_thresholds[metric] = find_dist_threshold( X_val, y_val)
            self.kNNClf_ratioThresholds[metric] = find_probaRatioThreshold( X_val, y_val)
            
    
    # Predict function with threshold
    def predict_kNNClf(self,  X, metric ='l2'):
        """
        X: np.array(1,d) : feature embedding of the current face to classify 

        Returns : tupl: faceNames prediction, prediction index, prediction probability   
        
        """
        model = self.kNNClfs[metric] # pipeline
        #prediction = model.predict(X)  #   array (1,)  of int64
        predict_probas = model.predict_proba(X)[0] # array (1, Nlabels) of int64  
        predictions_sorted = sorted(enumerate(list(predict_probas)),
                                    key=lambda x: x[1], 
                                    reverse=True)
      
        #print(predictions_sorted) # list of tuples 
        
        if len(predictions_sorted) >1:
            pred0 = predictions_sorted[0][1]
            pred1 = predictions_sorted[1][1] + 0.001 # to avoid division by 0
            predict_ratio = pred0/pred1 
        else: 
            predict_ratio = 2
        predicted_index, pred_proba = predictions_sorted[0]

        #print(predictions_sorted[0]  )                       
        #print(predicted_index,  pred_proba)
                
                
        #s ad-hoc threshold :TODO to determine more systematically
        if predict_ratio < self.kNNClf_ratioThresholds[metric]:
            return 'stranger', pred_proba   # The less the proba, the more likely it is to be 'stranger'
        
        return self.faceNames[predicted_index], pred_proba 

    def testModel_kNNClf(self, X_test, y_test, clf_metric):
        # Test set performance
        y_pred = self.predict_kNNClf(X_test, clf_metric )
        print(classification_report(y_test, y_pred))
    
        
 
    # ==============================================================================================
        
    def createGUI_tkinter(self,img):
        """  We ask the user to identify the unknow faces in the new image in unknows_new directory"""
        
        title = 'Stranger identification: (UGLY GUI )'   # title of the window
        imgMsg = 'Unidentified !!'                       # Text on the image    
        questionMsg = f"Please, identify this stranger ? (If you can't let it blank.)"
        answerMsg = f"The stranger name is"

        label_font = 'Arial 11'
        
        # Create a Tkinter window
        root = tk.Tk()
        root.geometry("600x344")   
        root.title(title)

        def _return_face_name(user_input):
            """ The face name is either the user input or 'stranger_{next_numero}"""
            
            if user_input is not None and len(user_input)>0:
                return unidecode(user_input.lower())  # convert UTF-8 (e.g: accents... ) in ASCII 
            else:         
                # We count how many kinds of 'stranger' directories are already there.     
                countStrangers = len(fl.listStrangers())  
                return f'stranger_{countStrangers}' 


        def handle_user_input(faceImg    = img):
            """  This function uses the following variables that are in its scope:
            question_label, entry, entry_frame, message1_label, message2_label 
            """
            user_input = entry.get().strip()  
            print(f"You entered: {user_input}")                
                            
            face_name = _return_face_name(user_input)            
            question_label.pack_forget()  # Remove the question  
            entry_frame.pack_forget()  # Remove the frame containing the entry and button
            entry.unbind("<Return>")   # Unbind the Enter key (cannot trigger an event again)
                                 
            msg1 = f"Hi, we call you \'{face_name}\'."
            msg2 = f"From now, I will do my best to recognize the face of {face_name}." 
            message1_label.config(text=msg1)                
            message2_label.config(text=msg2)

            # We save the face image in the directory named face_name (which is created if needed)        
            fl.saveNewFaceImg(face_name,faceImg)
                                     
        
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
        label = tk.Label(root, image=img_tk)
        label.pack()

        # Create a label for the question
        question_label = tk.Label(root, text=questionMsg, font = label_font)
        question_label.pack( pady=5)

        # Create a frame for the entry widget and the button
        entry_frame = tk.Frame(root)
        entry_frame.pack()

        # Create an entry widget to capture user input
        entry = tk.Entry(entry_frame)
        entry.pack(side=tk.LEFT )  

        # Create a submit button
        submit_button = tk.Button(entry_frame, text="Done", command=handle_user_input)
        submit_button.pack(side=tk.LEFT, padx=5)

        # Pressing the button change these message labels
        message1_label = tk.Label(root, text="", font = label_font)
        message1_label.pack()
        message2_label = tk.Label(root, text="", font = label_font)
        message2_label.pack()

        # Bind the Enter key to the handle_user_input function
        entry.bind("<Return>", on_enter)

        # Tkinter event loop
        root.mainloop()

    
       
            
    ## =========  tests   ==========================================
    async def test_runTask(self):
        """To test the asyncio/process/threads structure
        Without doing anything but passing numbers as data"""
        while True:
            faceImg = await asyncio.get_event_loop().run_in_executor(self.executor, 
                                                                    self.inputQueue.get)
            await asyncio.sleep(0.1)  # simulate processing time
        
            result = 'stranger' # faceImg
            print(f"Task is done: result: {result}")
            await self.resultQueue.put(result) 




    
