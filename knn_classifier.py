

""" 
knn_classifier.py

"""

import numpy as np

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report





class KnnClassifier:
    
    def __init__(self,X, y, faceNames: list):
   
        # -----------------------------------------------------------------------------                
        #      k-NearestNeigbor for classification of the face features embeddings
        self.clfs = {'l2':None, 'cosine':None}  #  Dictionary of KNeighborsClassifier 
        #  n_neighbors: to be determine via cross-validation
        # some possibily useful methods : get_params(), set_params()
        #                                 kneighbors()  : find the k neighbors and their distances  
        #                                 predict(), predict_proba(X)      
        # scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
     
        self.isTrained = False
        
        self.prepareKnnClassifier(X, y)
    # =========================  Preparation ============================================      

    def load(self):
        #TODO
        NotImplemented  
      
    def prepareKnnClassifier(self, X, y):    
        # Compute or load the knn classifier (self.predict_kNNClf[metric] )   
        if not self.isTrained:    
            self.train(X, y)   # compute KNN classifiers
            self.isTrained = True
        else: 
            self.load()    
                
        
    # ======== k-nearest-neighbors classifier to apply on the top of the SFace features embeddings ======    
    def train(self,X, y):
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
            self.clfs[metric] = grid_search.best_estimator_ 
            
            model = self.clfs[metric]['knn']  # type : KNeighborsClassifier
            print( f'k-nearest-neighbor with k={model.n_neighbors} and metric={metric}')

    def predictKNN(self,  X, metric ='l2'):
        """
        X: np.array(1,d) : feature embedding of the current face to classify 

        Returns : tupl: faceNames prediction, prediction index, prediction probability   
        
        """
        model = self.clfs[metric] # KNeighborsClassifier object
        
        #prediction = model.predict(X)  #   array (1,)  of int64
        predict_probas = model.predict_proba(X)[0] # array (1, Nlabels) of int64  
        predictions_sorted = sorted(enumerate(list(predict_probas)),
                                    key=lambda x: x[1], 
                                    reverse=True)
      
        predicted_index, pred_proba = predictions_sorted[0]               
        #return self.faceNames[predicted_index], pred_proba 
        return predicted_index, pred_proba
    
    # =================================================================================
    # ==TODO  Not sure...  looks more like data exploration than a test !!! What do it do with it ? ========      
    def testModel(self, X_test, y_test, clf_metric):
        # Test set performance
        y_pred = self.predictKNN(X_test, clf_metric )
        print(classification_report(y_test, y_pred))
    
          
  