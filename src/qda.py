#!/usr/bin/env python3

import yaml
import pickle
import rospy
from processing_cvsa.msg import features
from rosneuro_msgs.msg import NeuroOutput
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

class Qda:
    def __init__(self):
        rospy.init_node('qda', anonymous=True)
        yaml_file = rospy.get_param('~path_qda_decoder') 
        self.configure(yaml_file)
  
        rospy.Subscriber('/cvsa/features', features, self.callback)
        self.pub = rospy.Publisher('/cvsa/neuroprediction', NeuroOutput, queue_size=10)
        
        rospy.spin()
        
    def configure(self, path_qda_file):
        with open(path_qda_file, 'r') as file:
            params = yaml.safe_load(file)
        
        # Create a new QDA model
        self.qda = QuadraticDiscriminantAnalysis()
        
        # Set the parameters
        self.qda.priors_ = np.array(params['QdaCfg']['params']['priors'])
        self.qda.means_ = np.array(params['QdaCfg']['params']['means'])
        self.qda.rotations_ = np.array(params['QdaCfg']['params']['rotations'])
        self.qda.scalings_ = np.array(params['QdaCfg']['params']['scalings'])
        self.qda.covariance_ = np.array(params['QdaCfg']['params']['covs'])
        self.qda.classes_ = np.array(params['QdaCfg']['params']['classlbs'])
        
        # save parameters to extract the correct features
        self.bands_features =  np.array(params['QdaCfg']['params']['band'])
        self.idchans_features = np.array(params['QdaCfg']['params']['idchans']) - 1 # how matlab save
    
    def extract_features(self, msg):
        data = msg.data
        all_bands = np.array(msg.bands).reshape(-1, 2)
        nrows = len(all_bands)
        ncols = len(data) // nrows
        
        # Reshape the data
        reshaped_data = np.array(data).reshape(nrows, ncols)
        
        
        # Concatenate the features
        dfet = []
        for i, c_band_features in enumerate(self.bands_features):
            for j, filter_band in enumerate(all_bands):
                if np.array_equal(c_band_features, filter_band):
                    dfet.append(reshaped_data[j, self.idchans_features[i]])
                    break  # Exit the inner loop once a match is found
        
        return dfet
        
    def callback(self, msg):
        
        dfet = self.extract_features(msg)
        
        dfet = np.array(dfet).reshape(1, -1)
        prob = self.qda.predict_proba(dfet)
        
        output = NeuroOutput()
        output.header.stamp = rospy.Time.now()
        output.softpredict.data = prob[0].tolist()
        self.pub.publish(output)
        
   
    

if __name__ == '__main__':
    Qda()
