# -*- encoding: utf-8 -*
"""
filtering.py
"""
import numpy as np
import cv2 as cv
import json



import file_management as fl 
import visualization as v

"""
 We use the Kalman Filter already implemented in OpenCV : 
 https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html

"""
 
class Filtering:
    lastPrediction =None      # Class variable, accessible from any instance
    
    def __init__(self, modeLabel, initPt=None ):
        """ modeLabel   :  string in ['detection', 'tracking']
            initPt      : tuple (x0,y0) = initial coordinate of the measurements
        """
        
        # Filtering.measurements =/= Trajectory.observations because of formats
        # Filtering.measurements & predictions: np.float32 for KalmanFilter to operate
        # observations & filteredObs ( from Trajectory)
        # to adjust according to the coordinate format required by the microcontroller  
        
    
        self.measurements = list()
        self.predictions  = list()

        R, q_x, q_v, dt, a, b = self._returnKalmanParam(modeLabel)
        self.kalman = self._createKalmanFilter(R, q_x, q_v, dt, a, b)  
         
        if initPt is not None or Filtering.lastPrediction is not None:        
            self.setKalmanInitialState(*initPt) # initPt = (x0,y0) 
            
        else: 
            print(f'''The kalman filter for trajectory in mode={modeLabel} has been created, 
                  but no initial point has been set yet. ''')
            
        
    def _returnKalmanParam(self, mode):
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
        measurCov = self.computeMeasurementNoise()
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


        
    def _createKalmanFilter(self, R, q_x, q_v, dt=1, a=1, b=1):    
        """ Create the Kalman filter for a simple inertial model:
        Does not initialize the kalman filter states: kalman.statePre & kalman.statePost
        
        Args: 
            R : np.array : measurement noise covariance that needed to be pre-determined
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
       
        """  Explanation:
        Posterior Error Covariance. updated error covariance after incorporating the measurement. 
        It is updated during the correction step.
        """
        # Param a, b : default 1
        kalman.errorCovPost = np.array([[a, 0, 0, 0], 
                                        [0, a, 0, 0], 
                                        [0, 0, b, 0], 
                                        [0, 0, 0, b]], np.float32)
        
        """ Explanation: 
        Posterior Error Covariance
        you typically initialize errorCovPost to an identity matrix 
        or some other reasonable guess, errorCovPre is automatically calculated during the predict step 
        based on the errorCovPost and the process noise covariance. However, it's important to ensure that errorCovPost is initialized correctly because it influences the initial errorCovPre.
        
        Prior Error Covariance. predicted error covariance before incorporating the measurement. 
        It is calculated during the prediction step.
        """
        #kalman.errorCovPre = np.eye(4, dtype=np.float32) *0.05
        return kalman
    
        
    def setKalmanInitialState(self, x0=None, y0=None, vx0=0, vy0=0):
        """Define initial state estimate"""
        if x0 is None and y0 is None :
            if Filtering.lastPrediction is not None: 
                x0, y0, vx0, vy0 = Filtering.lastPrediction         
            else: 
                print('The Kalman Filter cannot be instantiated. An initial state is missing.')
        initialState=np.array([x0, y0, vx0, vy0], np.float32).reshape(-1, 1)
        self.kalman.statePre = initialState
        self.kalman.statePost = initialState.copy()
 
        
    def update(self, newMeasurement):
        """  
        Arg   :  newMeasurement
        inputs:  kalmanFilter, predictions """
        prediction = self.kalman.predict()
        newMeasurement = self._convertInFloat32(newMeasurement)
        self.kalman.correct(newMeasurement) # 
        self.predictions.append(prediction.copy())   # list of arrays of shape (4,2,1)
        self.measurements.append(newMeasurement)
        self.setLastPrediction(prediction)
        

    def _convertInFloat32(self, newMeasurement):
        # TODO: rester consistent dans le choix de type (float32 vs int16 etc)
        # int16 has to be transformed into float32 for the kalman filter to work on it        
        return np.array(newMeasurement, np.float32).reshape(2,1) 
         
       
    def computeMeasurementNoise(self):
        """ Returns the covariance matrix of the measurement process 
            for each mode (detection | tracking)
        
        Returns 
            dict {'detection': np.array[np.float32], 'tracking': np.array[np.float32]}
        """     
        jsonFile = ['faceCenterTraject_0_tracking.json',
                    'faceCenterTraject_0_detection.json']
        
        t,x,y = fl.openTraject(jsonFile[0])
        
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

    # ------- class methods ( by contrast with instance methods)-----------------------
    def setLastPrediction(self, prediction):
        Filtering.lastPrediction = prediction
        #print(Filtering.lastPrediction)
        
    def getLastPrediction(self):
        return Filtering.lastPrediction
    
        
# ==========  Estimation of the noise covariance to use in the Kalman filter ==================

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
 
              
#  ============= Tests & simple examples =============================================
import matplotlib.pyplot as plt

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

    x,y,t = fl.openTraject(jsonFile[0])
    
     
    # Filtering initialization
    #kalmanFilter,predictions= self.initFiltering((x[0], y[0]),mode='tracking')
    #measurements =[]
    trackingFilter = Filtering((x[0], y[0]))
    for i in range(len(t)):
        newMeasurement = getMeasurement(i, x, y)   # (2,1)
    
        trackingFilter.update(newMeasurement)
     
    v.showTraj(trackingFilter.measurements, trackingFilter.predictions)

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