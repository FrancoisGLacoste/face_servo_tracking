# -*- encoding: utf-8 -*-

"""
face_recognition.py 
"""
import os, time 
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue as qu
import multiprocessing as mp

import cv2 as cv
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

import file_management as fl
import util as u
import new_face_gui_tk as gui

"""  ====================== SFACE ======================================= 
SFace : [Zhonyu2021] a state-of-the-art algorithm for face recognition
Ref:
https://github.com/zhongyy/SFace/blob/main/SFace_torch/train_SFace_torch.py
https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py#L99

======================================================================="""


class FaceRecognition:
    
    def __init__(self,isActive = True, faceNames= None):
        self._isActive = isActive  

        self.faceNames = self.returnFaceNames(faceNames) # face names we want to recognize
         
        self.inputQueue = mp.Queue()       # Queue for moving face images to the face recognition task
        # WARNING *** !! multiprocessing queue are slow when transferring large objects !! 
        # (images are large objects that needed to be serialized)
        # https://www.mindee.com/blog/why-are-multiprocessing-queues-slow-when-sharing-large-objects-in-python
        
        self.resultQueue = asyncio.Queue() # Queue for capturing the recognition results (i.e. face names)
        self.executor = ThreadPoolExecutor(max_workers=2) # Thread pool for the face recognition task        
        
        self.recognizer = cv.FaceRecognizerSF.create(
                            os.path.join(fl.BASE_DIR,'face_recognition_sface_2021dec.onnx'), 
                            "" )
        
        self.faceEmbeddingsDict ={name: list() for name in self.faceNames} 
        self.X = None 
        self.y = np.array(range(len(self.faceNames))) # class labels are indices in faceNames
        
        # ------------ For dist-based discrimination of the unrecognized category -------
        # TODO ?  Should we organize this criterion into a new class ???
        self.centroidsDict   = {name: None for name in self.faceNames}  
        self.discreteDistribDict = {'l2': dict(), 'cosine':dict()} 
        self.radiusDict      = {'l2': dict(), 'cosine':dict()}
        self.kernelDensity   = {'l2': dict(), 'cosine':dict()}
        
        # ------------   Classifiers to apply to the face embeddings ------------------------- 
        self.LogisticRegressionClf = LogisticRegression(random_state=0)
        self.LogRegr_isTrained = False
        
        # --- k-NearestNeigbor for classification of the face features embeddings-------
        # k-nearest-neighbors classifiers
        self.kNNClfs = dict()  #  Dictionary of kNN classifiers with respect to various metrics
        #  n_neighbors: to be determine via cross-validation
        # some possibily useful methods : get_params(), set_params()
        #                                 kneighbors()  : find the k neighbors and their distances  
        #                                 predict(), predict_proba(X)      
        # scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        
        self.kNN_isTrained = False
        # ------------------ For new face identification  --------------------------------------
        self.newFaceIdThread = ThreadPoolExecutor(max_workers=1) # Thread for running newFaceId 
        # Queue for moving a new face identity (faceName) from  newFaceIdGUI
        self.newFaceIdQueue = qu.Queue() 
        self.newFaceIdGUI = None # Only created when required for new face identification
        self.idTime = 0
    
      
    def afterNewFaceId(self) : 
        # 
        timeAfterNewFaceId =  time.time()  -  self.idTime
        if timeAfterNewFaceId < 120 : # 1 minute
            return True
              
    def isActive(self ):
        
        # When a new face has just been id: dont ask for recognition again for some time    
        if not self.afterNewFaceId():
            return self._isActive  
        
            
    
    def cropBoxes(self,img, boxes, inGray=False):  
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
            
    def sendToFaceRecognition(self, img, faces ):
        """ 
        img  :    UMap, cv2.typing.MatLike or np.array : the whole camera frame (image)
        faces:    np.array [:,:4] :  a sequence of boxes, one box for each face    """
        #Rem: detection output: faces is array[(faceNb,15)]: array of faceNb faces
        # TODO:  devrais mettre cette fct dans une classe ??
        face_imgs = self.cropBoxes(img, faces)   
        '''recognizer.alignCrop(src_img: cv2.typing.MatLike, face_box: cv2.typing.MatLike, aligned_img: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
            
            recognizer.alignCrop(src_img: UMat,               face_box: UMat,               aligned_img: UMat               | None = ...) -> UMat: ...
        '''
        print('Sending image to face recognition module.')
        # Put the face image in a (non-async) queue for face recognition
        self.putImgQueue(face_imgs)    
    

    def putImgQueue(self, img):
        """ Put the face image in the queue for face recognition. 
        The image will show up in runFaceRecognitionTask()        
        """
        self.inputQueue.put(img)

    # ================== For the execution of face recognition loop ===============          
    async def runFaceRecognitionTask(self):
        """ *** This loop should have a high priority by comparison with 
        the retrieveResults() loop that ask the user to identify the unrecognized face 
        through the GUI. 
        This loop should continue to run while the retrieveResults loop is waiting for a 
        response for the user... 
        """
        print('we arrive in runFaceRecognitionTask')
        while True:
            # Receive a face image from the face detection loop
            print('We are awaiting for the last face image that has been detected.')
            faceImgs = await asyncio.get_event_loop().run_in_executor(self.executor,self.inputQueue.get)
            print('We got a new face image for the face recognition task to process.')
                
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
                print(f"Result received")
                self.resultQueue.task_done()
                           
                for img, name,_ in result:
                    if img is not None:
                        if name == 'unrecognized': 
                            # Ask the user to id the new face 
                            # The answer is sent to retrieveNewFaceId to be processed
                            gui.createGUI_tk(img, self.newFaceIdQueue) 
                            self.guiTime = time.time()  #TODO: in GUI class
                            
                            # TODO: convert the GUI into an object
                        
            except asyncio.CancelledError as e:
                print('The task has been canceled, exit the loop')
                break
            except Exception as e:
                # Handle any other exceptions that might occur
                print(f"Error in run() function: {e}")
                # Notify the queue that the task is done
                self.resultQueue.task_done()

    async def processNewFaceId(self):
    
        while True:
    
            print('We are awaiting for the new face id.') # for an non-async queue sending data to newFaceIdThread
            newFaceId = await asyncio.get_event_loop().run_in_executor(self.newFaceIdThread,
                                                                    self.newFaceIdQueue.get)
            faceName, faceImg = newFaceId
            
            print(f'We got {faceName}   to process.')
            print(type(faceImg))

            # The image is saved in the directory 'name' (which is created if needed)        
            fl.saveNewFaceImg(faceName, faceImg )
        
            # Update face embeddings for the new face name
            self.updateFaceEmbedding(faceName, faceImg)
 
            # Re-train the kNN classifier with the updated data
            
            # (?)TODO  For now it is silly: we simply convert all face embeddings again ! 
            # (but I think it is fast enough to justify avoiding the complications... )    
            self.stackFaceEmbeddingsInArray()  # recompute and set self.X
            self.trainKNNClfs(self.X, self.y)
            print(f'We just retrained the kNN layer to (hopefully) recognize {faceName}.')

            self.prepareUnrecognitionCriteria() 
            print('We just recomputed the distance-based criteria for unrecognized faces.')
            
    # =========================  Preparation ============================================      

    def loadkNNClfs(self):
        #TODO
        NotImplemented  
    
    def returnFaceNames(self,faceNames =None, exclude=None):
        if faceNames is None:
            #Default: list face_names i.e. directory names in DATAPATH, except the ones that end with '_new'
            return fl.listFaceNames() # ex: ['audrey', 'francois', 'victor', ']
        else:
            return faceNames 
      
    def prepareFaceRecognition(self): 
        
        # Compute or load self.faceEmbeddingsDict[face_name] 
        if not self.embeddingsAreSaved():
            self.computeAllFaceEmbeddings() 
        else: 
            self.loadEmbeddings()
        
        # Compute or load the knn classifier (self.kNNClfs[metric] )   
        if not self.kNN_isTrained:    
            self.stackFaceEmbeddingsInArray()  # sets self.X, self.y
            self.trainKNNClfs(self.X, self.y)
            self.kNN_isTrained = True
        else: 
            self.loadkNNClfs()  
        
        # TODO:Compute or load eventual additional classifiers (logistic regression...)
        
        
        # prepare the distance threshold criteria for recognition
        self.prepareUnrecognitionCriteria()        
        print('FaceRecognition preparation has been done.')      
    
    # =========================    Face feature embeddings ==============================    
    def computeFaceEmbeddings(self, face_names):
        """Train SFace algorithm by computing face embeddings.
        
        It prepares a dictionary of the embeddings of face features of each face name
        self.faceEmbeddingsDict = { face_name1:  array(N0, dim), 
                                    face_name2:  array(N1, dim),
                                    ...}
                                    where   Nn = number of images 
                                            dim = embedding dimension (=128)
        """
        for face_name in face_names:
            embeddings = []
            for face_img in fl.yieldFaceImgs(face_name):     #np.ndarray
                # REM: ils utilisaient aligncrop() pour obtenir face_img
                embedding = self.recognizer.feature(face_img)
                #print(embedding.shape)   # (1,128)
                dim = embedding.shape[1]
                embeddings.append(embedding)
            
            Nn = len(embeddings)   
            self.faceEmbeddingsDict[face_name]  = (np.array(embeddings)).reshape(Nn, dim)
            
        
    def computeAllFaceEmbeddings(self):
        """ 'Train' SFace algorithm by computing the face embeddings 
           for all face names that exists in DATAPATH   
        """
        if self.isActive:
            self.computeFaceEmbeddings(self.faceNames)
            print('We have just computed all face embeddings.')
        else:
            print('Face recognition is inactive: no embedding computed.')
            
                
    def updateFaceEmbedding(self, faceName, faceImg): 
        """  Updates face embeddings with the last new face image that have just been 
        identified by the user. 
        
        """
        # Compute the feature embedding of the new face image
        X = self.recognizer.feature(faceImg)   # (1,128)  
    
        # In case we never met 'faceName' before:
        if faceName not in self.faceEmbeddingsDict.keys(): 
            self.faceEmbeddingsDict[faceName] = X
        else:         
            # stack them
            X_n = self.faceEmbeddingsDict[faceName]  # (Nn, 128)
            self.faceEmbeddingsDict[faceName] = np.vstack([X,X_n]) # (Nn+1, 128)
        print('We just add the face embeddings to the new {faceName} image.')
            
    
    '''# TODO: A ENLEVER
    def convertFaceEmbeddingInArrays_1faceName(self, n, faceName):
        """  Convert the faceEmbeddings in arrays for a given faceName associated to index n"""
        print(self.faceEmbeddingsDict[faceName].shape)
        X_n =self.faceEmbeddingsDict[faceName]
        #X_n = np.array(self.faceEmbeddingsDict[faceName])  # array(Nn, 128)  where Nn = #{images}
        lenght = X_n.shape[0]
        #X_n = X_n.reshape(lenght,128)
        y_n = (np.array([n]*lenght)).reshape(lenght,)      # array(Nn, )
        return X_n, y_n
    '''
    
    def stackFaceEmbeddingsInArray(self):
        """   
        Face Embeddings is a dictionary of array of image feature embeddings: 
                            {'audrey': array(N0,128), 'francois':array(N1,128)}
                            where img_embed1, img_embed2, ...are np.array(1,d) , d=128 dimensions
            
        But sci-kit-learn methods expect the dataset to be in np.arrays
        i.e. 
        X : np.array(Nn, dim) : data points (i.e.faceEmbeddings) for a given label (i.e. face name)
        y : np.array(Nn,)     : array of label indices ( i.e. each index represents a face name)
        """
        for n, faceName in enumerate(self.faceEmbeddingsDict.keys()):
            #embed_list = self.faceEmbeddingsDict[faceName] # list of array(1,128)
            #y_n = n*len(embed_list)
            X_n = self.faceEmbeddingsDict[faceName]
            Nn = X_n.shape[0]
            y_n = (np.array([n]*Nn)).reshape(Nn,)      # array(Nn, )
            
            # Stack the arrays X <--- [ X|X_n ]
            if n == 0:
                X = X_n
                y = y_n 
            else:
                X = np.vstack([X,X_n])
                y = np.hstack([y,y_n])
                
        #return X,y
        self.X = X
        self.y = y 
 
    # TODO ?        
    def embeddingsAreSaved(self):
        """ 
        Returns a Boolean value: True when we find a file that contains the SFace features embeddings 
        
        """
        # TODO
        return False
    
    def loadEmbeddings(self):
        # TODO
        NotImplemented
        
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
    
        
    async def recognizeFace(self, newFaceImg):
        """ 
        Returns tuple (predicted_faceName, prediction_probability)
        
        We compute it as the faceName associated with the metric whose predictor 
        has the best predicted_proba      
        """
        newFace_features = self.recognizer.feature(newFaceImg) # X
        
        try:        
            # i.e. {metric : (faceName, predict_proba) for metric in [l2,cosine]}  
            predict_proba = { metric:  self.predict_kNNClf(newFace_features, metric ) 
                                for metric in ['l2', 'cosine']
                            } 
        
            faceNames_proba_list = list(predict_proba.values()) # [(faceName0, proba0), (faceName0, proba0)]
            print(faceNames_proba_list)
            
            # faceName such that proba is max in [(faceName_l2, proba_l2), (faceName_cos, proba_cos)]
            (faceName, faceName_prob), index = u.argmax_tupls(faceNames_proba_list )
            metric = ['l2', 'cosine'][index] # metric giving the best result above 
            
            print(f'We recognize the face of {faceName} with prob={faceName_prob}')
            
            # False if unrecognized, according to the distance threshold criterion
            if not self.isRecognized(newFace_features,faceName,metric):
                print('Finally the face is classified as unrecognized.')
                return 'unrecognized', None
            print(f'Cassified as recognized: {faceName} ')   
        except IndexError as ie: 
            print('in recognizerFace', ie)
            faceName = 'unrecognized'
            faceName_prob = 1
        except Exception as e:
            print(e)
            faceName = 'unrecognized'
            faceName_prob = 1
              
        finally: 
            return faceName, faceName_prob
             
        
        
    def trainKNNClfs(self,X, y):
        """  Trains k-nearest-neighbors classifiers
        
        X: np.array (n, d):   n data points in d-dimensions
        y: list len(y)=n , OR np.array (1,n): data labels   
        """        
        '''
        def find_probaRatioThreshold(X_val, y_val):
            """Returns the optimal threshold for the ratio between first ans second nearest classes. 
            
            The closer to 1 is the ratio, the lower is the classification signicance. i.e the actual
            classification might be 'stranger'. 
            """
            threshold=2
            return threshold 
        '''        
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

    # ========== dist-based Threshold criteria for the unrecognized category ============
    
    def dist_l2(self,features1, features2=None):  
        features1 = features1.reshape(1,128).astype(dtype=np.float32)
        if features2 is None: 
            features2 = np.zeros(features1.shape)
        features2 = features2.reshape(1,128).astype(dtype=np.float32) 
        dist1 = self.recognizer.match(features1, features2,cv.FaceRecognizerSF_FR_NORM_L2)
        #dist2 = np.linalg.norm(features1 - features2, ord=2).astype(dtype=np.float32)  # OK dist2 == dist1
        #dist3 = np.sum((features1 - features2)**2).astype(dtype=np.float32)   # square of l2 dist
        return dist1   # dist2 and np.square(dist3)  are equivalent 
    
    def dist_cosine(self,features1, features2=None):
        features1 = features1.reshape(1,128).astype(dtype=np.float32)
        if features2 is None: 
            features2 = np.zeros(features1.shape)
        features2 = features2.reshape(1,128).astype(dtype=np.float32)
        return self.recognizer.match(features1, features2,cv.FaceRecognizerSF_FR_COSINE)     
        
 
                        
    def computeCentroids(self):
        for faceName in self.faceEmbeddingsDict.keys():
            # faceEmbeddingsDict : list of arrays
            xArr =self.faceEmbeddingsDict[faceName] #  array (Nn, 128)            
            Nn = xArr.shape[0]
            dim =  xArr.shape[1] # =128
            centroid = (np.sum(xArr,0)/Nn).reshape(1, dim)
            self.centroidsDict[faceName] = centroid.astype(dtype=np.float32)
                
       
    def computeDistToCentroid(self, X, faceName, dist ='l2'):
        """Compute the distance dist of X to the centroid of faceName   """
        distFct = {'l2':self.dist_l2 ,'cosine': self.dist_cosine }
        centroid = self.centroidsDict[faceName] 
        
        return distFct[dist](centroid, X)
 
               
    def computeDistDistrib(self):
        """ 
        For each faceName and each distance metric (l2 or cosine):
        Compute the discrete distribution of the distances to the centroid. 
        and the radius (=maximum distance between the centroid and a data point)
        i.e.: 
        It sets:  
            self.distDistribDict[dist][name] : list of increasing distance values
            self.radiusDict[dist][name]      : maximum distance
        """                  
        for dist, distFct in [('l2',self.dist_l2), ('cosine',self.dist_cosine)]:
            for name in self.faceEmbeddingsDict.keys():
                centroid =  self.centroidsDict[name]
                X_n = self.faceEmbeddingsDict[name]
                #print(type(centroid))
                #print(centroid.shape)
                distrib = sorted([distFct(centroid,x) for x in X_n])
                self.discreteDistribDict[dist][name] = distrib
                
    def computeRadius(self):
        for dist in ['l2','cosine']:
            for name in self.faceEmbeddingsDict.keys():
                distrib =self.discreteDistribDict[dist][name]
                self.radiusDict[dist][name] = distrib[-1]            
                        
    

    def estimateDistDensity(self, discreteDistrib):
        # Not Necessarily useful but can be interesting to understand the data
        """ 
        Estimate the density distribution of distances to centroid 
        with kernel density (unsupervised learning)
        Only makes sense when Np large enough (Np > Np_thresh )
        
        discreteDistrib : list  (like self.distDistribDict[dist][name] )
        """
        
        Np_thresh = 8
        for dist in ['l2','cosine']:
            for name in self.faceEmbeddingsDict.keys():
                distancesList = self.discreteDistribDict[dist][name]
                
                Np = len(distancesList)
                if Np >= Np_thresh:
                    X = np.array(distancesList)  # array of distances
                    
                    # TODO Choice of the smoothing parameter: 
                
                    
                    h=0.2   
                    density = KernelDensity(kernel ='gaussian', bandwidth=h).fit(X)
                    self.kernelDensity[dist][name] =density 
       
    def prepareUnrecognitionCriteria(self):
        
        # Criteria is based on a distance threshold and centroids 
        # for all faceName      
        self.computeCentroids()   # Centroids of the data clusters 
        self.computeDistDistrib() #   
        self.computeRadius()      # Radius of the data clusters   
        
                
    def isRecognized(self,X,faceName,dist='l2'):
        """
        Return: 
            True when 'recognized' and we keep the {faceName} prediction of the classifier (kNN...)
            i.e.  if the distance of X to centroid < the maximal distance to centroid (i.e. the radius)
         
            False when 'unrecognized'          
         """        
        if self.computeDistToCentroid(X, faceName, dist) < self.radiusDict[dist][faceName]:
            return True
        else: 
            return False
        
    # =============================    ===============================
       
        
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
      
        predicted_index, pred_proba = predictions_sorted[0]               
        return self.faceNames[predicted_index], pred_proba 

    def testModel_kNNClf(self, X_test, y_test, clf_metric):
        # Test set performance
        y_pred = self.predict_kNNClf(X_test, clf_metric )
        print(classification_report(y_test, y_pred))
    
        
 
    ## ============================== For (face features) data visualization ===========================         
     
    def project3D_PCA(self):
        from sklearn.decomposition import PCA   # Bad practice !! temporary
        
        X = self.convert
        pca = PCA(n_component =3,whiten=False )
        data = pca.fit_transform(X)
    
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




    
