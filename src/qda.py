#!/usr/bin/env python3

import yaml
import pickle
import rospy
from processing_bci.msg import eeg_power
from rosneuro_msgs.msg import NeuroOutput
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

class Qda:
    def __init__(self):
        rospy.init_node('qda', anonymous=True)
        self.qda_name = "qda_model"
        
        try:
            self.path_decoder = rospy.get_param('~path_qda_model')
        except KeyError as e:
            rospy.logfatal(f"[{self.qda_name}] Mandatory parameter: 'path_qda_model'. Error:{e}.")
            return
        
        topic_sub = rospy.get_param('~topic_sub', '/eeg_power')
        
        try:
            self.qda_paradigm = rospy.get_param('~qda_paradigm')
        except KeyError as e:
            rospy.logfatal(f"[{self.qda_name}] Mandatory parameter: 'qda_paradigm'. Error:{e}.")
            return
        self.qda_name += f"_{self.qda_paradigm}"
        
        conf = self.configure()
        if not conf:
            rospy.logfatal(f"[{self.qda_name}] Error in the QDA configuration.")
            return
        else:
            rospy.loginfo(f"[{self.qda_name}] QDA configurated correctly.")


        rospy.Subscriber(topic_sub, eeg_power, self.callback)
        self.pub = rospy.Publisher(f'/{self.qda_paradigm}/neuroprediction/raw', NeuroOutput, queue_size=10)
        
        rospy.spin()
        
    def configure(self):
        
        try:
            with open(self.path_decoder, 'r') as file:
                params = yaml.safe_load(file)
        except Exception as e:
            rospy.logerr(f"[{self.qda_name}] Error loading QDA YAML file: {e}")
            return False
        
        try:
            self.qda_name = yaml.safe_load(open(self.path_decoder, 'r'))['QdaCfg']['name']
        except Exception as e:
            rospy.logerr(f"[{self.qda_name}] Error loading QDA name: {e}")
            return False    
        
        try:
            qda_params = params['QdaCfg']['params']
        except Exception as e:
            rospy.logerr(f"[{self.qda_name}] Error getting the QDA's parameters structure: {e}")
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
            raw_idchannels = qda_params['idchannels']
            if raw_idchannels and isinstance(raw_idchannels, list) and not isinstance(raw_idchannels[0], (list, tuple)): # here just for the cvsa case (a single list instead of a list of lists)
                raw_idchannels = [raw_idchannels]
            self.idchans_features = [np.array(ids) - 1 for ids in raw_idchannels]
            self.nfeatures = int(qda_params['nfeatures'])
            self.nclasses = int(qda_params['nclasses'])
        except Exception as e:
            rospy.logerr(f"[{self.qda_name}] Error getting the QDA's parameter: {e}")
            return False

        return True
    
    def extract_features(self, msg):
        data = msg.data
        nchannels = msg.nchannels
        nbands = msg.nbands
        all_bands = np.array(msg.bands).reshape(-1, 2)
        
        reshaped_data = np.array(data).reshape(nbands, nchannels) # [bands x channels]
        
        if len(reshaped_data) == 0:
            rospy.logerr(f"[{self.qda_name}] No matching bands found between features and incoming data.")
            return
        
        dfet = [] 
        for i, c_band_features in enumerate(self.bands_features):
            for j, filter_band in enumerate(all_bands):
                if np.array_equal(c_band_features, filter_band):
                    c_channels_idx = self.idchans_features[i]
                    for idx_ch in c_channels_idx:
                        dfet.append(reshaped_data[j, idx_ch])
                    break 
                
        if(len(dfet) != self.nfeatures):
            rospy.logerr(f"[{self.qda_name}] Error in the feature extraction: expected {self.nfeatures} features, but got {len(dfet)}.")
            return
                
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
        output.decoder.classes = self.qda.classes_.astype(int).tolist()
        self.pub.publish(output)
        

if __name__ == '__main__':
    Qda()
