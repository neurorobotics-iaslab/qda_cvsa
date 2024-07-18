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
        self.subject = rospy.get_param('~subject')
        yaml_file = yaml_file + '/cfg/qda_' + self.subject + '.yaml'

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
        self.idchans_features = np.array(params['QdaCfg']['params']['idchans']) - 1
        
    def extract_features(self, msg):
        all_bands = np.array(msg.bands)
        all_bands = [all_bands[i:i+2].tolist() for i in range(0, len(all_bands), 2)]
        data = np.reshape(msg.data, (len(all_bands), len(msg.data)//len(all_bands)))
        
        # take the bands of interest
        dfet = []
        for i in range(0, len(self.bands_features)):
            c_band_features = self.bands_features[i]
            for j in range(0, len(all_bands)):
                if all(c_band_features == all_bands[j]):
                    dfet.append(data[j][self.idchans_features[i]-1]) ######## TODO: with gdf place -1 in the channels selection
                    
        return dfet
        
    def callback(self, msg):
        
        dfet = self.extract_features(msg)
        
        dfet = np.array(dfet)
        dfet = dfet.reshape(1, -1)
        prob = self.qda.predict_proba(dfet)
        
        output = NeuroOutput()
        output.header.stamp = rospy.Time.now()
        output.softpredict.data = prob[0].tolist()
        self.pub.publish(output)
        
   
    

if __name__ == '__main__':
    Qda()
