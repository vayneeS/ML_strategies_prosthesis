import random
import time 
import pickle
import datetime
import numpy as np
import matplotlib.pylab as plt
import gesture_classifier 
import myo_manager
from time import perf_counter

# num_samples = 300

def populate_gestures(num_trials=5):
    g_map = {'Wrist Pronation':0,'Wrist Supination':1, 'Hand Closing':2, 'Hand Opening':3, 'Pinch Closing':4, 'Pinch Opening':5,'Rest':6}
    #g_map = {'Wrist Pronation':0}
    gesture_set =[]
    for k in g_map.keys():
        gesture_set.extend([k] * num_trials)
    g_rand = random.sample(gesture_set, len(gesture_set))
    return g_rand,g_map

def get_gesture(g_rand):
    #print(len(g_rand))
    #return g_rand.pop()
    return

def get_emg_data(listener,hub,gesture,num_samples):
    period_ms = 100
    res = []
    print(len(res))
    start_time = time.time()
    print("\nGesture -- ", gesture, " : Ready?")
    input("Press Enter to continue...")
    while hub.run(listener.on_event, duration_ms=period_ms):
        print(listener.emg_data)

        for emg in listener.emg_data:
            if (len(res)<num_samples):
                res.append(emg)
        pickle_data(res)
        listener.clear_emg()
        break    
        
    print(res)
    return np.array(res)

# def get_emg_data(listener, hub, num_samples):
#     verification_set = np.zeros((num_samples, 8))
#     t1_start = perf_counter()
#     while True:
#         #try:
#         hub.run(listener.on_event, 20000)
#         #time.sleep(3)
#         data_array = listener.get_data_array()
#         # print(data_array)
#         if(len(data_array) > 0):
#             verification_set = data_array[0]
#             print('num samples:', len(verification_set))
#             #print(verification_set)
#             pickle_data(verification_set)
#         data_array.clear()
#         break
#         #except:
#             #print("exception occurred")
#         t1_stop = perf_counter()
#         print("Elapsed time:",  t1_stop-t1_start)
#     return verification_set

def get_emg(gestures, listener,hub):
    """
    Returns the emg data recorded. Each gesture is recorded num_trials number of times
    , in a random order, for 5 seconds. With a sample rate of 25 ms, there are approx 
    200 samples per trial 
    
    Args:
        gestures (list):   randomly sorted list of strings, repeated num_trials times
        listener (myo.DeviceListener): listens to Myo device events
    
    Returns:
        emg_data (dictionary): Emg data arrays for each gesture
    """
    emg_data = {}
    duration_emg = 2000
    freq_emg = 200
    num_samples = (duration_emg / 1000) * freq_emg
    for g in gestures:
        if(g not in emg_data.keys()):
            emg_data[g] = []
        print("\nGesture -- ", g, " : Ready?")
        input("Press Enter to continue...")
        start_time = time.time()
        t1_start = perf_counter()
        
        while hub.run(listener.on_event, 200):
            emg_data_tmp = listener.on_update_emg()
            print(emg_data_tmp)
            for emg in emg_data_tmp:
                emg_data[g].append(emg)
                print('num samples: ',len(emg_data[g]))              
                if time.time() - start_time > 5.0:
                    break
            
        t1_stop = perf_counter()
        print("Elapsed time:",  t1_stop-t1_start)
        
    pickle_data(emg_data,gestures)
    return emg_data

def pickle_data(emg_data):   
    """
    pickles emg data and gestures into file 
    """  
    curr_time = datetime.datetime.now()
    fname = 'emg_data'+str(curr_time)+'.pkl'
    #flog = 'gestures_log'+str(curr_time)+'.pkl'
    pickle.dump(emg_data, open(fname, 'wb'))
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
    feat_zc = []
    feat_ar = []
    feat_mn = []
    #for k in emg_data.keys():
        #data = np.array(emg_data[k])
    rms = f.rms(emg_data)
    feat_rms.extend(rms) 
    #feat_mn.extend(f.win_mean(emg_data))
    #feat_zc.extend(f.zc(data))
    #feat_ar.extend(f.ar(data,4))
    features = np.array(feat_rms)
    #features = np.concatenate((features, feat_zc, feat_ar), axis=1)
    labels.extend([map[gesture]] * len(rms))
    labels = np.array(labels)
    return features,labels

def train_model(algo,features,labels):
    model = gesture_classifier.Classifier(algo)
    # feats = []
    # for k in features.keys():
    #     f_temp = np.reshape(features[k],(-1,8))
    #     feats.extend(f_temp) 
    model.train_validate(features,labels)

    return model

def plot_emg(data):
    #for k in emg_data.keys():
        #data = emg_data[k]
        _, axis = plt.subplots(8, 1)
        # for arr in range(len(data)):
        #     m_ges.extend(data[arr])
        #m_ges = np.reshape(data,(-1,8))
        print(data.shape)
        min_c=np.min(data)
        max_c=np.max(data)
        y_lim = (min_c,max_c)
        plt.ylim([min_c, max_c])
        plt.setp(axis, ylim=y_lim)
        for c in range(data.shape[1]):
            axis[c].plot(data[:,c])
        axis[-1].set_xlabel('samples')
        #axis[3].set_ylabel(k)
        
        plt.show()

if __name__ == '__main__':
    l = myo_manager.Listener()
    l.set_sdk()
    hub = l.get_hub()
    g_rand,g_map = populate_gestures(5)
    for g in g_rand:
        data = get_emg_data(l,hub,g,300.0)
        features,labels = get_features_labels(data,g_map,g)
        plot_emg(features)

    #data = open_pickle('data/emg_data2022-08-23 14:59:24.721015.pkl')
   # features,labels = get_features_labels(data,g_map)
#     # plot_emg(data)
#     train_model('KNN',features,labels)