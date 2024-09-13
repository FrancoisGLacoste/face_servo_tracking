# -*- encoding: utf-8 -*
"""
filtering.py
"""
import numpy as np
import cv2 as cv
import json
import matplotlib.pyplot as plt

GREEN = (10,255,0)
BLUE = (255,0,0)
RED =  (0,0,255) 
YELLOW = (50,200,200)
MAGENTA=(255, 0, 255)
CYAN = (255,255,0)
BLACK = (0,0,0)

projetPath = '/home/moi1/Documents/dev_py/vision/PROJET_Face-Tracking-camera/'

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

"""
 Kalman Filter In OpenCV : 
 https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html

"""

# ==============================================================================================
"""
Kalman Filter in dlib 
dlib.net/dlib/filtering/kalman_filter.cpp.html
dlib.net/dlib/filtering/kalman_filter.h.html

class kalman_filter
class momentum_filter:  http://dlib.net/python/index.html#dlib_pybind11.momentum_filter

Kalman filter with a state transition model of:

    x{i+1} = x{i} + v{i} 
    v{i+1} = v{i} + some_unpredictable_acceleration

and a measurement model of:

    y{i} = x{i} + measurement_noise




momentum_filter find_optimal_momentum_filter 

"""

        
def visualizeTraject(img, trajectory, color=GREEN):
    """
    Draw the trajectory of the non-filtered face center coordinates upon the video frame image,
    Args:
        img :_type_: image, frame from the video
        faceCenterTraject :list of (int16,int16) coordinates: 
                                Trajectory of the non-filtered face center coordinates

    Returns:
        _type_: image from the video with an additional trajectory drawn upon it
    """
    try:
        for p in trajectory:
            x = int(np.round(p[0]))
            y = int(np.round(p[1]))
            cv.circle(img, (x,y), 1, color, 1)
    #except # just skip if fails
    
    finally:    
        return img

def saveTraject(faceCenterTraject, faceCenterTraject_t, mode):
    n=1
    # If faceCenterTraject is long enough, we save it in JSON to represent
    # the non-filtered signal behavior in the case of each mode 
    # (detection mode and tracking mode)  

    if len(faceCenterTraject) > 20:
        with open(projetPath+ f'faceCenterTraject_{n}_{mode}.json', 'w') as f:
            data = {'coord': faceCenterTraject, 
                    'time':faceCenterTraject_t}
            json.dump(data, f, cls=NpEncoder)
 
  
def openTraject(file):
    """We open and plot the trajectory To visualize how we should filter it."""
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

def plotTraject(x,y, t, x_label = "t     [seconds]" ):
    fig, axs = plt.subplots(2)#(figsize=(10, 6))
    fig.suptitle('Fitting face trajectory in tracking mode')
    #axs[0].plot(t, x)
    #axs[1].plot(t, y)
    #ax.set_title("Kalman Filter in 2D Motion Estimation")
    
    for n,z in enumerate([x,y]):
        axs[n].scatter(t, z, c='red', s=1, label=f"Noisy Observations in {z}")
    #ax.plot(predictions[:, 0], predictions[:, 1], 'b-', label="Kalman Filter Estimation")
    axs[1].set_xlabel(x_label)
    #ax.set_ylabel("x")
    #ax.set_xlim(0, 10)
    #ax.set_ylim(-1.5, 1.5)
    #ax.legend()


    #----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Fitting face trajectory in tracking mode')
    #ax.plot(t, x)   
    ax.scatter(t, x, c='red', s=1, label=f"Noisy Observations in x")
    ax.set_xlabel(x_label)
    #----

    plt.show() 

#================================================================

#-------sans sklearn ni statsmodels ----------
from scipy.stats.distributions import norm    
def estimNoise(x,y, t=None) :
    def manual_cov(x):    
        m = np.mean(x)
        return np.sum((x - m) * (x - m)) / (len(x) - 1)        
    
    def np_cov(x):
        return  np.cov(x)
    
    
    # autocovariance estimation
    def cov_n(x, n=0):
        m = np.mean(x)
        shiftx = x[-n:]+ x[:-n]   #circular shift x by n
        return np.array((shiftx)-m).dot(np.array(x)-m) /(len(x) -1)
       

    #def statsmodels_cov(x):
    #    return sm.cov_struct.Covariance(x).cov
   
   
    for n,c in enumerate([manual_cov, np_cov, cov_n ]):
        for m,z in enumerate([x, y]):
            print(f'cov{n}( x_{m} ) = ', c(z))
    print('--------------')
    """
    ProcessMLEResults.covariance(time, scale, smooth)
    
    Must have len(time) rows.
    Scale and smooth should be data arrays whose columns align with the fitted scaling and smoothing parameters.
    """
 
#------------------  avec sklearn ------------- 
from sklearn.covariance import empirical_covariance 
def estimNoise_sklearn(x, y, t=None) :
    """
    """
    # https://scikit-learn.org/stable/modules/covariance.html#covariance
    #Maximum likelihood covariance estimator

    #print(x)
    #print(y)
    x_y = np.array(list(zip(x,y)), np.float32) # list of lists
    #print( x_y.transpose() )
    
    """X:  ndarray of shape (n_dim, n_samples) """ 
    cov_x_y = empirical_covariance(x_y)
    return cov_x_y # ndarray of shape (n_dim, n_dim)
  
def returnKalmanParam(mode):
    """_summary_

    Args:
        mode (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
        
        
        
   
    For face tracking (not face detection ): 
    Parameters : 
    dt =  1.2  # np.mean(dt_list[:-1]/dt_list[1:])  # 1.13
    
    ProcessNoise Covariance: 
    q_x=1e-4
    q_v = 1e-3  # roughly between 8e-4 and 4e-3
        #q_v = 1e-1 # overfit and not smooth enough
        #q_v = 1e-2 # # better fit but not smooth enough
        #q_v = 5e-3 # +/- OK: still good fit, but slightly not smooth enough
        #q_v = 1e-3 # not bad , good trade-off
        #q_v = 8e-4 #  not bad , good trade-off
        #q_v = 6e-4 # still the same problems than 1e-4, but less pronounced
        #q_v = 1e-4 #smoother but bad fit when high acceleration /deccelaration
        # the best would be to adapt parameters in case of abrupt speed changes 
        # Check dlib library
      
    measurementNoiseCov :  estimated by MLE with sklearn empirical_covariance fct
    On stationary data: 
    R =  [[1.1656 0.09  ]
          [0.09   0.79  ]]
    On full non-stationary trajectory: 
    
    Posterior Error Covariance : typicaly identity matrix 
    a=1 # better than 1e-1
    b=1 # better than b=2  also better than 1e-1
         
    """
    #jsonFile = ['faceCenterTraject_0_tracking.json',
    #        'faceCenterTraject_0_detection.json']

    #x,y,t = openTraject(projetPath+jsonFile[0])
    
    # dict {'tracking': ..., 'detection': }
    measurCov = returnMeasurNoiseFromTraj()
    R = measurCov[mode]
    
    #dt_list = np.diff(t)
    #dt =  np.mean(dt_list) # il faudrait adapter la matrice au pas de temps
    #print(dt) # 0.082
    dt =  1.2#np.mean(dt_list[:-1]/dt_list[1:]) 
    
    q_x=1e-4
    #q_v = 1e-1 # 
    #q_v = 1e-2 # 
    q_v = 1e-3 # 
    #q_v = 8e-4 #  
    #q_v = 6e-4 # 
    #q_v = 1e-4 #smoother but bad fit when high acceleration /deccelaration
    # the best would be to adapt parameters in case of abrupt speed changes 
    # Check dlib library
    
    # --- errorCovPost -----
    a=1
    b=1 # better than b=2 and also better than 1e-1   
    return R, q_x, q_v, dt, a, b


    
def createKalmanFilter(x0, y0, R, q_x, q_v, dt=1, a=1, b=1):    
    """ Create and initialize the Kalman filter
    
    Args: 
        R : 
    """
    
    
    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                            [0, 1, 0, 0]], 
                                            np.float32)
    kalman.transitionMatrix = np.array([[1, 0, dt, 0], 
                                        [0, 1, 0, dt], 
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]], np.float32)

    # q_x, q_v : param   , default: 1e-4   
    kalman.processNoiseCov = np.array([
                                    [q_x, 0, 0, 0], 
                                    [0, q_x, 0, 0], 
                                    [0, 0, q_v, 0], 
                                    [0, 0, 0, q_v]], np.float32)

    # Define measurement noise covariance
    kalman.measurementNoiseCov = R.astype(np.float32)
    
    
    # Define initial state estimate and initial estimate error covariance
    initialState=np.array([x0, y0, 0, 0], np.float32).reshape(-1, 1)
    kalman.statePre = initialState
    kalman.statePost = initialState.copy()
    
    """  Explanation:
    Posterior Error Covariance. updated error covariance after incorporating the measurement. 
    It is updated during the correction step.
    """
    # Param a, b : default 1
    kalman.errorCovPost = np.array([[a, 0, 0, 0], 
                                    [0, a, 0, 0], 
                                    [0, 0, b, 0], 
                                    [0, 0, 0, b]], np.float32)
    
    """
    Posterior Error Covariance
    you typically initialize errorCovPost to an identity matrix 
    or some other reasonable guess, errorCovPre is automatically calculated during the predict step 
    based on the errorCovPost and the process noise covariance. However, it's important to ensure that errorCovPost is initialized correctly because it influences the initial errorCovPre.
    
    Prior Error Covariance. predicted error covariance before incorporating the measurement. 
    It is calculated during the prediction step.
    """#kalman.errorCovPre = np.eye(4, dtype=np.float32) *0.05
    
    return kalman
    
     
def returnMeasurNoiseFromTraj():
    """ Returns the covariance matrix of the measurement process 
        for each mode (detection | tracking)
    
    Returns 
        dict {'detection': np.array[np.float32], 'tracking': np.array[np.float32]}
    """     
    jsonFile = ['faceCenterTraject_0_tracking.json',
                 'faceCenterTraject_0_detection.json']
    
    t,x,y = openTraject(projetPath+jsonFile[0])
    
    # Initial state condition:
    #print(f'x0={x[0]}')
    #print(f'y0={y[0]}')
    
    # Stationary  trajectory : 
    x_station = x[10:110]
    y_station = y[10:110]
    t_station = t[10:110]
    # As reference, to compared with the MLE
    #estimNoise(x_station,y_station) 
    
    # covariance matrix of the measurement process
    R =estimNoise_sklearn(x_station,y_station).astype(np.float32) 
    
    #print(type(R[[0]]))
    #print(R.shape)
    R_detect =R
    R_track = R #TODO
    measurCov= {'detection': R_detect,
                'tracking': R_track}
    return measurCov

def showTraj(measurements, predictions):
    measurements = np.array(measurements).squeeze()
    predictions = np.array(predictions)
    predictions = predictions.squeeze()

    # sample number
    nb =predictions[:, 0].shape[0]  
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 700)
    ax.set_ylim(250, 450)
    ax.set_title("Kalman Filter in 2D Motion Estimation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
                
    # Plotting the noisy measurements and Kalman filter estimations
    ax.scatter(measurements[:, 0], 
                measurements[:, 1], 
                c='red', s=10, label="Noisy Measurements")
    ax.plot(predictions[:, 0].reshape((nb,1)), 
            predictions[:, 1].reshape((nb,1)),
            'b-', label="Kalman Filter Estimation")
    
    #ax.scatter(predictions[:, 0].reshape((nb,1)), 
    #    predictions[:, 1].reshape((nb,1)), 
    #    c='blue', s=10, label="Kalman Filter Estimation")

    ax.legend()        
    plt.show() 

def initFiltering(initPt, mode='tracking'):
    #TODO: to refactor Filtering  into a class
    # kalmanFilter, measurements, predictions : attributes
    # init(), update()  : methods
    x0, y0 = initPt
    R, q_x, q_v, dt, a, b = returnKalmanParam(mode)
       
    kalmanFilter = createKalmanFilter(x0, y0, R, q_x, q_v, dt, a, b)  
    #measurements = []
    predictions  = []
    return  kalmanFilter, predictions
 
def updateFiltering(newMeasurement, kalmanFilter, predictions):
    prediction = kalmanFilter.predict()
    kalmanFilter.correct(newMeasurement) # ****PLANTE ICI dans face_servo_tracking ***
    predictions.append(prediction.copy())   # list of arrays of shape (4,2,1)
    return predictions
 
         
#-----   Tests & simple examples
def testKalmanFilter():
    """    
    Try to filter face center trajectories in the visual field
    with linear Kalman filter
    The initial covariance matrix is estimated via MLE (sklearn)
    or estimated via the usual sample estimator 
    
    """     
    
    # To simulate the data stream : 
    def getMeasurement(i, x, y):
        """ return points sequentially from a trajectory""" 
        return np.array([[x[i]],[y[i]]], np.float32)        
 
    jsonFile = ['faceCenterTraject_0_tracking.json',
                'faceCenterTraject_0_detection.json']

    x,y,t = openTraject(projetPath+jsonFile[0])
    
     
    # Filtering initialization
    kalmanFilter,predictions=initFiltering((x[0], y[0]),mode='tracking')
    measurements =[]
    for i in range(len(t)):
        newMeasurement = getMeasurement(i, x, y)   # (2,1)
    
        predictions = updateFiltering(newMeasurement,kalmanFilter, 
                                                   predictions)
        measurements.append(newMeasurement) # type: list of arrays of shape (2,1)
    showTraj(measurements, predictions)

def simpleExample():
    """ 
   
    The simplest example :  
    but the filtered signal is significantly shifted (lag) even 
    if the model and the noises are perfectly known. 
    """
    # Generate synthetic data: sin(i) + Gaussian noise
    num_points = 100
    true_signal = np.sin(np.linspace(0, 2 * np.pi, num_points))
    measurements = true_signal + np.random.normal(0, 0.1, num_points)

    # Kalman Filter setup
    kf = cv.KalmanFilter(2, 1)  # 2 dynamic parameters, 1 measurement

    # State vector [position, velocity]
    kf.statePre = np.array([[0], [0]], dtype=np.float32)

    # State transition matrix A
    dt = 1  # Assume time step is 1 for simplicity
    kf.transitionMatrix = np.array([[1, dt], [0, 1]], dtype=np.float32)

    # Measurement matrix H
    kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)

    # Process noise covariance Q
    kf.processNoiseCov = np.array([[1e-5, 0], [0, 1e-5]], dtype=np.float32)

    # Measurement noise covariance R
    kf.measurementNoiseCov = np.array([[0.1]], dtype=np.float32)

    # Initial state covariance P
    kf.errorCovPost = np.array([[1, 0], [0, 1]], dtype=np.float32)

    # Store the results
    filtered_measurements = []

    for measurement in measurements:
        # Correct step
        kf.correct(np.array([[measurement]], dtype=np.float32))
        # Predict step
        prediction = kf.predict()
        filtered_measurements.append(prediction[0, 0])

    # Plot the results
    plt.plot(true_signal, label='True signal')
    plt.plot(measurements, label='Noisy measurements')
    plt.plot(filtered_measurements, label='Kalman filtered')
    plt.legend()
    plt.show()
   

                    
if __name__ =='__main__':
    #simpleExample()
    testKalmanFilter()