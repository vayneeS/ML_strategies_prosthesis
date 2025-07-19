from __future__ import print_function
import myo
import random
from time import perf_counter
import pickle
import datetime
import gesture_classifier
import numpy as np
import matplotlib.pyplot as plt

class Emg(myo.DeviceListener):

  def __init__(self):
    super(Emg, self).__init__()
    self.emg = []

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    self.emg = event.emg

def get_emg(gesture,num_samples):
    listener = Emg()
    samples = []
    myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')
     
    try:
        hub = myo.Hub('com.vaynee.myo-python')
        period_ms = 50
        print("\nGesture -- ", gesture, " : Ready?")
        #input("Press Enter to continue...")
        
        t1_start = perf_counter()
        while hub.run(listener.on_event, period_ms):
            if(len(listener.emg)>1):
                samples.append(listener.emg)
            if(len(samples)==num_samples):
                break
        pickle_data(samples,gesture)
        t1_stop = perf_counter()
        print("Elapsed time:",  t1_stop-t1_start)
        print(len(samples))
    except:  
        print("Unable to connect to Myo Connect. Is Myo Connect running?")
    
    return samples

def populate_gestures(num_trials=5):
    g_map = {'Wrist Pronation':0,'Wrist Supination':1, 'Hand Closing':2, 'Hand Opening':3, 'Pinch Closing':4, 'Pinch Opening':5,'Rest':6}
    #g_map = {'Wrist Pronation':0}
    gesture_set =[]
    for k in g_map.keys():
        gesture_set.extend([k] * num_trials)
    g_rand = random.sample(gesture_set, len(gesture_set))
    return g_rand,g_map

def pickle_data(data,gesture):   
    """
    pickles emg data and gestures into file 
    """  
    curr_time = datetime.datetime.now()
    fname = gesture+str(curr_time)+'.pkl'
    #flog = 'gestures_log'+str(curr_time)+'.pkl'
    pickle.dump(data, open('data/'+fname, 'wb'))
    #pickle.dump(gestures,open('myo_scripts/data/'+flog,'wb'))

def open_pickle(path):
    """
    opens pickle file named file_name
    """
    with (open(path, "rb")) as f:
        while True:
            try:
                data = (pickle.load(f))
            except EOFError:
                break
    return data

def get_features_labels(emg_data,map,gesture):
    """returns num_windows x 8 features and num_windows labels
    """
    f = gesture_classifier.Features()
    labels = []
    feat_rms = [] 
    feat_iav = []
    feat_zc = []
    rms = f.rms(emg_data)
    feat_rms.extend(rms) 
    feat_iav.extend(f.iav(emg_data))
    feat_zc.extend(f.zc(emg_data))
    #plot_emg(feat_zc)
    features = np.array(feat_rms)
    features = np.concatenate((features, feat_iav, feat_zc), axis=1)
    labels.extend([map[gesture]] * len(rms))
    labels = np.array(labels)
    return features,labels
    

def train_model(algo,features,labels):
    model = gesture_classifier.Classifier(algo)
    model.train_validate(features,labels)
    return model

def plot_emg(data):
    #for k in emg_data.keys():
        #data = emg_data[k]
        _, axis = plt.subplots(8, 1)
        #fig,axis=plt.subplots(8, 1, figsize=(12, 3), sharey=True)
        # for arr in range(len(data)):
        #     m_ges.extend(data[arr])
        #m_ges = np.reshape(data,(-1,8))
        data = np.array(data)
        print(data.shape)
        min_c=np.min(data)
        max_c=np.max(data)
        y_lim = (min_c,max_c)
        plt.ylim([min_c, max_c])
        plt.setp(axis, ylim=y_lim)
        for c in range(data.shape[1]):
            axis[c].plot(data[:,c])
        axis[-1].set_xlabel('window')
        #axis[3].set_ylabel(k)
        
        plt.show()

if __name__ == '__main__':
    listener = Emg()
    all_features = []
    all_labels = []
    g_rand,g_map = populate_gestures(2)
    data_files = ['Hand Closing2022-09-13 17:11:40.364206.pkl','Hand Closing2022-09-13 17:14:08.872856.pkl','Hand Opening2022-09-13 17:11:21.953167.pkl','Hand Opening2022-09-13 17:13:50.220193.pkl','Pinch Closing2022-09-13 17:10:26.288719.pkl','Pinch Closing2022-09-13 17:13:11.915749.pkl','Pinch Opening2022-09-13 17:11:03.623754.pkl','Pinch Opening2022-09-13 17:12:36.054569.pkl','Rest2022-09-13 17:12:53.549053.pkl','Rest2022-09-13 17:14:26.963230.pkl','Wrist Pronation2022-09-13 17:10:44.741770.pkl','Wrist Pronation2022-09-13 17:13:30.825255.pkl','Wrist Supination2022-09-13 17:11:58.675522.pkl','Wrist Supination2022-09-13 17:12:17.171816.pkl']
    for f in data_files:
        data = open_pickle('data/'+f)
        #print(data)
        str_ges = f[0:-30]
        features,labels = get_features_labels(np.array(data),g_map,str_ges)
        print('shape feats: ',features.shape)
        all_features.extend(features)
        all_labels.extend(labels)    

    train_model('KNN',all_features,all_labels)
    # for g in g_rand:
    #     data = get_emg(listener,g,300)
    #     features,labels = get_features_labels(np.array(data),g_map,g)
        
    #     print('shape feats: ',features.shape)
    #     all_features.extend(features)
    #     all_labels.extend(labels) 
