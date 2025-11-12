#!/usr/bin/env python3

import yaml
import pickle
import rospy
from processing_cvsa.msg import eeg_power
from rosneuro_msgs.msg import NeuroOutput
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

class Qda:
    def __init__(self):
        rospy.init_node('qda', anonymous=True)
        try:
            self.path_decoder = rospy.get_param('~path_qda_model')
        except KeyError as e:
            rospy.logfatal(f"Parametro mancante: {e}. Assicurati di lanciarlo con un launch file.")
            return
        conf = self.configure()
        if not conf:
            rospy.logfatal("Erorr in the QDA configuration.")
            return
        else:
            rospy.loginfo("QDA configurated correctly.")
  
        rospy.Subscriber('/cvsa/eeg_power', eeg_power, self.callback)
        self.pub = rospy.Publisher('/cvsa/neuroprediction/qda', NeuroOutput, queue_size=10)
        
        rospy.spin()
        
    def configure(self):
        
        try:
            with open(self.path_decoder, 'r') as file:
                params = yaml.safe_load(file)
        except Exception as e:
            rospy.logerr(f"Error loading QDA YAML file: {e}")
            return False
        
        try:
            self.qda_name = yaml.safe_load(open(self.path_decoder, 'r'))['QdaCfg']['name']
        except Exception as e:
            rospy.logerr(f"Error loading QDA name: {e}")
            return False    
        
        try:
            qda_params = params['QdaCfg']['params']
        except Exception as e:
            rospy.logerr(f"Error getting the QDA's parameters structure: {e}")
            return False
        
        # Create a new QDA model
        self.qda = QuadraticDiscriminantAnalysis(reg_param= qda_params['reg_param'])
        
        # Set the parameters
        try:
            self.qda.priors_ = np.array(qda_params['priors'])
            self.qda.means_ = np.array(qda_params['means'])
            self.qda.classes_ = np.array(qda_params['classes'])
            self.qda.rotations_ = np.array(qda_params['rotations'])
            self.qda.scalings_ = np.array(qda_params['scalings'])
            self.qda.covariance_ = np.array(qda_params['covs'])
            
            # save parameters to extract the correct features
            self.bands_features =  np.array(qda_params['bands'])
            self.idchans_features = np.array(qda_params['idchannels']) - 1 # matlab starts from 1 not 0
            self.nfeatures = int(qda_params['nfeatures'])
            self.nclasses = int(qda_params['nclasses'])
        except Exception as e:
            rospy.logerr(f"Error getting the QDA's parameter: {e}")
            return False

        return True
    
    def extract_features(self, msg):
        data = msg.data
        nchannels = msg.nchannels
        nbands = msg.nbands
        all_bands = np.array(msg.bands).reshape(-1, 2)
        
        reshaped_data = np.array(data).reshape(nchannels, nbands)
        
        dfet = [] 
        for i, c_band_features in enumerate(self.bands_features):
            for j, filter_band in enumerate(all_bands):
                if np.array_equal(c_band_features, filter_band):
                    c_channels_idx = self.idchans_features[i]
                    for idx_ch in c_channels_idx:
                        dfet.append(reshaped_data[idx_ch, j])
                    break 
                
        dfet = np.log(dfet) # apply the log transfromation    
         
        return dfet
        
    def callback(self, msg):   
        
        dfet = self.extract_features(msg)
        
        dfet = np.array(dfet).reshape(1, -1)
        probabilities = self.qda.predict_proba(dfet)[0]
        
        hard_pred_vector = np.zeros(self.nclasses, dtype=int)
        hard_pred_vector[np.argmax(probabilities)] = 1
        
        output = NeuroOutput()
        output.header.stamp = rospy.Time.now()
        output.neuroheader.seq = msg.seq
        output.softpredict.data = probabilities.tolist()
        output.hardpredict.data = hard_pred_vector.tolist() 
        output.decoder.type = self.qda_name
        output.decoder.path = self.path_decoder
        self.pub.publish(output)
        

if __name__ == '__main__':
    Qda()
