#!/usr/bin/env python3

# Node to test the extraction of features and the prediction of the QDA model, save the features and the QDA_value into 2
# different csv files

import yaml
import pickle
import rospy
from processing_cvsa.msg import features
from rosneuro_msgs.msg import NeuroOutput
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import csv

class Qda:
    def __init__(self):
        rospy.init_node('test_extraction', anonymous=True)
        yaml_file = rospy.get_param('~path_qda_decoder') 
        self.configure(yaml_file)
        
        rospy.Subscriber('/cvsa/features', features, self.callback)
        self.pub = rospy.Publisher('/cvsa/neuroprediction', NeuroOutput, queue_size=10)
        
        self.create_file = True
        
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
        self.idchans_features = np.array(params['QdaCfg']['params']['idchans']) - 1 #### keep this -1 for matlab
    
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
        
        all_bands = np.array(msg.bands)
        all_bands = [all_bands[i:i+2].tolist() for i in range(0, len(all_bands), 2)]
        data = np.reshape(msg.data, (len(all_bands), len(msg.data)//len(all_bands)))
        
        if self.create_file:
            mode = 'w'
            self.create_file = False
        else:
            mode = 'a'
        
        with open('/home/paolo/cvsa_ws/src/qda_cvsa/test/features_sended.csv', mode, newline='') as file:
            writer = csv.writer(file)
        
            # Write the data to the CSV file
            writer.writerow(msg.data)
            
        with open('/home/paolo/cvsa_ws/src/qda_cvsa/test/features_extracted.csv', mode, newline='') as file:
            writer = csv.writer(file)
        
            # Write the data to the CSV file
            writer.writerow(dfet)
            
        dfet = np.array(dfet).reshape(1, -1)
        prob = self.qda.predict_proba(dfet)
        
        output = NeuroOutput()
        output.header.stamp = rospy.Time.now()
        output.softpredict.data = prob[0].tolist()
        #output.softpredict.data = [0.5, 0.5]
        
        with open('/home/paolo/cvsa_ws/src/qda_cvsa/test/ros_probs.csv', mode, newline='') as file:
            writer = csv.writer(file)
        
            # Write the data to the CSV file
            writer.writerow(np.array(prob[0].tolist()))
        
        self.pub.publish(output)
        
   
    

if __name__ == '__main__':
    Qda()