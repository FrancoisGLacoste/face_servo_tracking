# -*- encoding: utf-8 -*-

"""
explore_data.py 

See doc:
PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Comparison PCA vs NCA vs LDA
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA , SparsePCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from sface_embeddings import SFaceEmbeddings
from file_management import listFaceNames
# ===========================================================================================
#      Data Preparation of SFace feature emnbeddings for our face dataset

def prepareFaceData():
    faceNames = listFaceNames()
    embeddings = SFaceEmbeddings(faceNames)
    embeddings.stackFaceEmbeddingsInArray()  

    # Global variables in this file
    X = embeddings.X
    y = embeddings.y 
    nClasses = len(embeddings.faceNames)
    return X, y, nClasses


##  ================================================================================================
##             Reduction of dimensionality :      PCA, SparcePCA         
##  ================================================================================================

def performFullPCA(X):
    """ Relevant doc from sci-kit-learn:
    PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, 
    iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None)
    
    *** n_components='mle' is only supported if n_samples >= n_features **** 
    
    PCA attributes: 
        components_   ndarray of shape (n_components, n_features)
        explained_variance_      ndarray of shape (n_components,)
        explained_variance_ratio_   ndarray of shape (n_components,)
        singular_values_    ndarray of shape (n_components,)
        mean_   ndarray of shape (n_features,)
       
    """
    N, D = X.shape
    print(f'Number of face samples: N={N}.')
    print(f'Sample dimension = number of features: D={D} (should be 128).')
    
    # For automatic choice of dimensionality , i.e. find n_components
    print('Begin to perform PCA.')
    pca = PCA(svd_solver='full')
    #pca.fit_transform(X)
    X_transf = pca.fit(X).transform(X)
    
    #print('explained_variance_ratio: ', pca.explained_variance_ratio_)  
    print('n_components_: ',pca.n_components_) # 110 dont 2 negligeables, donc 108
    #print('singular_values_: ', pca.singular_values_)   

    for pcNbr in [108, 100, 10, 3]:
        print(f'''The {pcNbr} first principal components (over 128) explain  
            {np.sum(pca.explained_variance_ratio_[:pcNbr])*100} % of the total variance.''')

    #X_transf[:, componentIndex]
    return X_transf, y

def performScaledPCA(X):
        N, D = X.shape
        print(f'Number of face samples: N={N}.')
        print(f'Sample dimension = number of features: D={D} (should be 128).')
        
        # For automatic choice of dimensionality , i.e. find n_components
        print('Begin to perform PCA.')
        #pca = PCA(svd_solver='full')
        
        #n_components=nDim,
        model = make_pipeline(StandardScaler(), PCA(svd_solver='full',))
        X_transf = model.fit(X).transform(X)
        
        return X_transf, y    
        
def performSparsePCA(X):
    """
    SparsePCA(n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, 
    tol=1e-08, method='lars', n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None)
    """ 
    spca = SparsePCA()
    NotImplemented   
 
    
##  ================================================================================================
##                   For data visualization in reduced spaces    
##  ================================================================================================

def comparePCA_LDA_NCA():
    n_neighbors = 2 # Hyperparameter k in k-nearest-neighbos method
    random_state = 0

    # Load Digits dataset
    X, y = prepareFaceData()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=random_state
    )

    dim = len(X[0])
    n_classes = len(np.unique(y))
    
    nDim =3
    # Reduce dimensionto {nDim}  with PCA
    pca = make_pipeline(StandardScaler(), PCA(n_components=nDim, 
                                              random_state=random_state))

    # Reduce dimension  with LinearDiscriminantAnalysis
    lda = make_pipeline(StandardScaler(), 
                        LinearDiscriminantAnalysis(n_components=nDim))

    # Reduce dimension  with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=nDim, 
                                       random_state=random_state),
    )

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Make a list of the methods to be compared
    dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()
        # plt.subplot(1, 3, i + 1, aspect=1)

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)

        # Plot the projected points and show the evaluation score
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y, s=30, cmap="Set1")
        plt.title(
            "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
        )
    plt.show()


def scatterPlot3D(fig,X_embedded, y):
    """    LA FENETRE SE FERME AUTOMATIQUEMENT EN SORTANT DE CETTE FCT
    https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

   
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y, s=30, cmap="Set1") #marker='m')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.show()
    #return ax
# ===========================================================================================
# TODO ??                         Density Estimation : in reduced spaces
# ===========================================================================================
# Problem is some classes have too few data points to allow a meaningful estimation
# Consequently, dimensionality reduction (as PCA) is needed before proceeding with density estimation
from face_recognition_SFace_oo_v3 import FaceRecognition

def estimateDistDensity(faceRecognition: FaceRecognition):    # TODO: smoothing hyperparameter ?
    # Not Necessarily useful but can be interesting to understand the data
    """ 
    Estimate the density distribution of distances to centroid 
    with kernel density (unsupervised learning)
    Only makes sense when Np large enough (Np > Np_thresh )
    
    discreteDistrib : list  (like self.distDistribDict[dist][name] )
    """
    
    Np_thresh = 8
    for dist in ['l2','cosine']:
        for name in faceRecognition.faceEmbeddingsDict.keys():
            distancesList = faceRecognition.discreteDistribDict[dist][name]
            
            Np = len(distancesList)
            if Np >= Np_thresh:
                X = np.array(distancesList)  # array of distances
                
                # TODO Choice of the smoothing parameter: 
            
                
                h=0.2   
                density = KernelDensity(kernel ='gaussian', bandwidth=h).fit(X)
                KernelDensity[dist][name] =density 

  
  
if __name__ == '__main__':
             
    X, y, n =prepareFaceData()
    
    # No scaling
    X_transf_no, y = performFullPCA(X)

    # With scaling before pca
    X_transf_scale, y = performScaledPCA(X)
    

    for X_transf in [X_transf_no, X_transf_scale]:
        #  3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_transf[:, 0], X_transf[:, 1], X_transf[:, 2], c=y, s=30, cmap="Set1") #marker='m')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        plt.show()