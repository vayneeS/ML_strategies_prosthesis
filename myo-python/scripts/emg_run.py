import myo_manager as m
import random
import time 
import pickle
import datetime
import numpy as np
import matplotlib.pylab as plt
import gesture_classifier as clf

def get_emg(gestures, listener):
    """
    Returns the emg data recorded. Each gesture is recorded num_trials number of times
    , in a random order, for 5 seconds. With a sample rate of 50 ms, there are approx 
    100 samples per trial 
    
    Args:
        gestures (list):   randomly sorted list of strings, repeated num_trials times
        listener (myo.DeviceListener): listens to Myo device events
    
    Returns:
        emg_data (dictionary): Emg data arrays for each gesture
    """
    hub =  listener.start_hub()
    emg_data = {}
    with open("gestures.json", "w") as f:
        for g in gestures:
            if(g not in emg_data.keys()):
                emg_data[g] = []
            print("\nGesture -- ", g, " : Ready?")
            input("Press Enter to continue...")
            start_time = time.time()

            while hub.run(listener.on_event, duration_ms=25):
                emg_data_tmp = listener.on_update_emg()
                for emg in emg_data_tmp:
                    emg_data[g].append(emg)
                #print(emg_data)
                if time.time() - start_time > 5:
                    break
            print('num samples: ',len(emg_data[g]))
    pickle_data(emg_data,gestures)
    return emg_data

def pickle_data(emg_data,gestures):   
    """
    pickles emg data and gestures into file 
    """  
    curr_time = datetime.datetime.now()
    fname = 'emg_data'+str(curr_time)+'.pkl'
    flog = 'gestures_log'+str(curr_time)+'.pkl'
    pickle.dump(emg_data, open(fname, 'wb'))
    pickle.dump(gestures,open(flog,'wb'))

def open_pickle(file_name):
    """
    opens pickle file named file_name
    """
    with (open(file_name, "rb")) as f:
        while True:
            try:
                data = (pickle.load(f))
            except EOFError:
                break
    return data

def get_features_labels(emg_data,map):
    f = clf.Features()
    labels = []
    feat_rms = []
    feat_zc = []
    feat_ar = []
    for k in emg_data.keys():
        data = np.array(emg_data[k])
        rms = f.rms(data)
        feat_rms.extend(rms) 
        feat_zc.extend(f.zc(data))
        feat_ar.extend(f.ar(data,4))
        features = np.array(feat_rms)
        features = np.concatenate((features, feat_zc, feat_ar), axis=1)
        labels.extend([map[k]] * len(rms))
    labels = np.array(labels)
    return features,labels

def train_model(algo,features,labels):
    model = clf.Classifier(algo)
    # feats = []
    # for k in features.keys():
    #     f_temp = np.reshape(features[k],(-1,8))
    #     feats.extend(f_temp) 
    model.train_validate(features,labels)

    return model

def plot_emg(emg_data):
    for k in emg_data.keys():
        data = emg_data[k]
        _, axis = plt.subplots(8, 1)
        # for arr in range(len(data)):
        #     m_ges.extend(data[arr])
        m_ges = np.reshape(data,(-1,8))
        min_c=np.min(m_ges)
        max_c=np.max(m_ges)
        y_lim = (min_c,max_c)
        plt.ylim([min_c, max_c])
        plt.setp(axis, ylim=y_lim)
        for c in range(m_ges.shape[1]):
            axis[c].plot(m_ges[:,c])
        axis[-1].set_xlabel('samples')
        axis[3].set_ylabel(k)
        
        plt.show()

if __name__ == '__main__':
    l = m.Listener()
    l.set_sdk()
    l.start_hub()
    num_trials = 5

    g_map = {'Wrist Pronation':0,'Wrist Supination':1, 'Hand Closing':2, 'Hand Opening':3, 'Pinch Closing':4, 'Pinch Opening':5,'Rest':6}
    keys =[]

    for k in g_map.keys():
        keys.extend([k] * num_trials)
    g_rand = random.sample(keys, len(keys))
    #data = get_emg(g_rand,l)
    data = open_pickle('emg_data2022-08-23 14:59:24.721015.pkl')
    features,labels = get_features_labels(data,g_map)
    # plot_emg(data)
    train_model('KNN',features,labels)