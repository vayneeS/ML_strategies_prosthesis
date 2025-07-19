
from __future__ import print_function
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import signal
import statistics as st
import matplotlib.pylab as plt
import time
import pickle
import sklearn.ensemble
from sklearn import metrics
import myo
from time import sleep
import numpy as np
import threading
import collections
import math
import random 
from sklearn.model_selection import GridSearchCV
import datetime
import MAB
import compute_features

class Listener(myo.DeviceListener):
    def __init__(self, n, queue_size=1):
        self.n = n
        self.lock = threading.Lock()
        self.ori_data_queue = collections.deque(maxlen=queue_size)
        self.emg_data_queue = collections.deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def on_connected(self, event):
        event.device.stream_emg(True)
        self.emg_enabled = True

    # def on_connect(self, timestamp, firmware_version):
     #   myo.set_stream_emg(myo.StreamEmg.enabled)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))
        # self.emg_data_queue = [event.timestamp, event.emg]

    def on_orientation_data(self, quat):
        # print("Orientation:", quat.x, quat.y, quat.z, quat.w)
        with self.lock:
            self.ori_data_queue.append(quat)

    def get_ori_data(self):
        with self.lock:
            return list(self.ori_data_queue)

    def on_update_emg(self):
        # print(self.on_emg)
        emg_data = self.get_emg_data()
        return emg_data


def get_emg(gesture):
    listener = Listener(500)
    print("\nGesture -- ", gesture, " : Ready?")
    input("Press Enter to continue...")
    emg_data = {}
    emg_data[gesture] = {}
    start_time = time.time()
    counter = 0
    while hub.run(listener.on_event, duration_ms=200):
        emg_data_tmp = listener.on_update_emg()
        # emg_data returns tuple ts_emg,8-valued array
        #if time.time() - start_time > 2:
            #counter += 1
        for i, ts_emg in enumerate(emg_data_tmp):
            if ts_emg[0] not in emg_data[gesture].keys():  # if timestamp is unique
                emg_data[gesture][ts_emg[0]] = ts_emg[1]
                
            #start_time = time.time()
        if time.time() - start_time > 5:
            break
        
        # if time.time() - start_time > 2:
        #     #print(time.time(),start_time)
        #     break

    train_data = np.array([emg_data[gesture][k]
                          for k in emg_data[gesture].keys()])
    #print(train_data.shape)

    max_sig = np.max(train_data)
    min_sig = np.min(train_data)
    thresh = min_sig + (max_sig-min_sig)/4
    #get electrode with max value 
    col = np.unravel_index(np.argmax(train_data, axis=None), train_data.shape)[1]
    #get max electrode's indices where values > thresh
    idx = np.where(train_data[:,col]>thresh)
    #print(train_data[idx,col])

    thresholded = []

    #filter unique rows
    for i in idx:
        thresholded.extend(train_data[i,:])

    thresholded = np.array(thresholded)
    #plt.plot(thresholded)
    #plt.show()
    print('thresh: ',thresholded.shape)
    return thresholded

def pickle_random_training_data(gest_map,num_trials,win_size,win_inc):
    train = {}
    log = []
    num_win = 0

    for j in range(num_trials):# reps
        lr = random.sample(gest_map.keys(), len(gest_map.keys()))

        for k in lr:# gesture string
            log.append(k)
            if(k not in train.keys()):
                train[k] = []
            while(num_win <= 0):
                data = get_emg(k)
                data_size = data.shape[0]
                num_win = math.ceil((data_size - win_size)/win_inc) + 1
            train[k].append(data)
            num_win = 0
            print(' ', k, 'trial:',j+1,' - size:', len(train[k]))
    curr_time = datetime.datetime.now()
    fname = 'random_training_data'+str(curr_time)+'.pkl'
    flog = 'log'+str(curr_time)+'.pkl'
    pickle.dump(train, open(fname, 'wb'))
    pickle.dump(log,open(flog,'wb'))
    return fname

def compute_sw(gest_map,n_s,x):
    sw = 0
    for i in gest_map.keys():#num classes
        m_i = np.mean(x[gest_map[i]][:], axis=0) #mean of all samples per class
        print(m_i)
        for j in range(n_s[i]):#num samples per class
            sw += (x[gest_map[i]][j] - m_i) * np.transpose(x[gest_map[i]][j] - m_i)
    return sw

def compute_sb(gest_map,n_s,x):
    sb = 0
    m = np.mean(x, axis=0)
    for i in gest_map.keys():#num classes
        m_i = np.mean(x[0:n_s[i]], axis=0) #mean of all samples per class
        sb += n_s[i]*(m_i - m)*np.transpose(m_i - m)
    return sb
#def select_gestures(gest_map,win_size,win_inc,bandit,algo):

def select_gestures(gest_map,win_size,win_inc,algo):
    num_samples = {}
    reward_dic = {}
    chosen_arm_vec = []
    nb_trials = 0
    all_feats = []

    pre_data = random_schedule('random_training_data2022-07-22 12:26:20.853524.pkl',gest_map,1,win_size,win_inc)
    clf,pre_data_feats_dic,pre_data_feats = get_model(pre_data,gest_map,win_size,win_inc,algo)
    i=0
    for k in pre_data_feats_dic.keys():
        reward_dic[k] = []
        all_feats.append([])
        class_data = np.array(pre_data_feats_dic[k][0])
        for arr in class_data:
            all_feats[i].append(arr)
        num_samples[k] = len(pre_data_feats_dic[k][0])
        i+=1
    all_feats = np.array(all_feats)
    
    sw = compute_sw(gest_map,num_samples,all_feats)
    #sb = compute_sb(gest_map,num_samples,all_feats)

    print('sw: ',sw)
    #print('sb: ', sb)
    # while(nb_trials < 30):
    #     gesture,reward = MAB.experiment(sw,sb,bandit)
    #     reward_dic[gesture].append(reward)
    #     chosen_arm_vec.append[gesture]
    #     data = get_emg(gest_map[gesture])
    #     clf,features = get_model_inc(data,gesture,win_size,win_inc,algo)#train classifier
    #     sw = compute_sw(gest_map,num_samples,all_feats)
    #     sb = compute_sb(gest_map,num_samples,all_feats)
    #     all_feats.extend(features)
    #     num_samples[gesture] += features.shape[0]
    #     nb_trials += 1

    return chosen_arm_vec

def open_pickle(file_name):
    with (open(file_name, "rb")) as f:
        while True:
            try:
                data = (pickle.load(f))
            except EOFError:
                break
    return data

def compute_feature(data,win_size,win_inc):
    num_signals = data.shape[1]

    data_size = data.shape[0]
    #print(data.shape)
    num_win = math.ceil((data_size - win_size)/win_inc) + 1        
    #print(num_win)

    feat = np.zeros((num_win, num_signals))

    start = 0
    end = win_size

    for i in range(num_win):
        cur_win = data[start:end, :]
        feat[i, :] = np.sqrt((np.square(cur_win)).mean(axis=0))
        start += win_inc
        end += win_inc
    return feat

def resample_features(original_data,features):
    for i in range(features.shape[1]):
        f = signal.resample(features[:,i], len(original_data[:,i]))
        xnew = np.linspace(0, len(original_data[:,i]), len(original_data[:,i]), endpoint=False)
        plt.plot(np.sqrt(np.square(original_data[:,i])),'-', xnew, f, '.-')
        plt.legend(['data', 'resampled'], loc='best')
        plt.show()


def train_validate_model(classifier,features,labels):
    accuracies = []
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)

    if(classifier == "LDA"):
        clf = LinearDiscriminantAnalysis(solver='svd')
        
    elif(classifier == "KNN"):
        clf = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)

    for train_index, test_index in skf.split(features, labels):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        #print(X_train.shape, X_test.shape, len(np.unique(y_train)), len(np.unique(y_test)))
        
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        # print(classifier.score(X_test, y_pred))
        plot_conf_mat(clf,y_test,X_test)
        #print(y_train)
        accuracies.append(accuracy_score(y_test, y_pred))
    print(np.mean(accuracies), np.std(accuracies))
    return clf

def plot_EMG(data):
    for k in data.keys():
        m_ges = []
        figure, axis = plt.subplots(8, 1)
        for arr in range(len(data[k])):
            m_ges.extend(data[k][arr])
        m_ges = np.array(m_ges)
        min_c=np.min(m_ges)
        max_c=np.max(m_ges)
        y_lim = (min_c,max_c)
        plt.ylim([min_c, max_c])
        plt.setp(axis, ylim=y_lim)
        for c in range(m_ges.shape[1]):
            axis[c].plot(m_ges[:,c])
        axis[-1].set_xlabel('time')
        axis[3].set_ylabel(k)
        
        plt.show()
     
def plot_conf_mat(clf,y_test,X_test):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    display_labels=clf.classes_)
    disp.plot()
    plt.show()

def plot_lda(X_train,y_train):
    lda = LinearDiscriminantAnalysis(solver='svd')
    d = lda.fit(X_train, y_train).transform(X_train)
    plt.figure(figsize=(15, 8))
    print('d: ',d.shape)
    print('X_train: ',X_train)
    plt.scatter(d[:,0],d[:,2],  c=y_train)#color = labels
    plt.show()

def random_schedule(fname,gest_map,num_trials,win_size,win_inc):
    if fname == '':
        fname = pickle_random_training_data(gest_map,num_trials,win_size,win_inc)
    return open_pickle(fname)

def get_model(data,gest_map,win_size,win_inc,algo):   
    #plot_EMG(data)
    features = []
    labels = []
    feats_ges = {}
    for k in data.keys():
        feats_ges[k] = []
        for arr in (data[k]):
            feats = compute_feature(arr,win_size,win_inc)
            feats_ges[k].append(feats)
            #feats is an array of arrays x by 8
            features.extend(feats)
            labels.extend([gest_map[k]] * len(feats))
            
    features = np.array(features)
    #plot_EMG(feats_ges)
    labels = np.array(labels)#num_win labels per class 
    clf = train_validate_model(algo,features,labels)
    return clf,feats_ges,features

def get_model_inc(data,ges,win_size,win_inc,algo):
    features = []
    labels = []
    for arr in (data):
        feats = compute_feature(arr,win_size,win_inc)
        features.extend(feats)
        labels.extend([ges] * len(feats))
            
    features = np.array(features)
    #plot_EMG(feats_ges)
    labels = np.array(labels)#num_win labels per class 
    clf = train_validate_model(algo,features,labels)
    return clf,features

def test_model(fname,clf,ges_map,num_trials,win_size,win_inc):
    if(fname == ''):
        fname = random_schedule('',gest_map,num_trials,win_size,win_inc)
    data = open_pickle(fname)
    features = []
    labels = []
    feats_ges = {}
    for k in data.keys():
        feats_ges[k] = []
        for arr in (data[k]):
            feats = compute_feature(arr,win_size,win_inc)
            feats_ges[k].append(feats)   
            features.extend(feats)
            labels.extend([ges_map[k]] * len(feats))
           
    features = np.array(features)
    #plot_EMG(feats_ges)
    labels = np.array(labels)
    X_test = features
    y_test = labels
    y_pred = clf.predict(X_test)
    print('y_test: ',y_test)
    print('y_pred: ',y_pred)
    
    accuracy =accuracy_score(y_test, y_pred)
    #Generate the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    print('CONF MAT')
    print(cf_matrix)
    print(accuracy)

if __name__ == '__main__':
    myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')
    hub = myo.Hub()
    feed = myo.ApiDeviceListener()
    gest_map = {'Wrist Pronation':0,'Wrist Supination':1, 'Hand Closing':2, 'Hand Opening':3, 'Pinch Closing':4, 'Pinch Opening':5,'Rest':6}
    win_size = 128
    win_inc = 8
    num_trials = 5
    feats_ges = {}
    labels = []
    labels_zc = [] 
    labels_rms = []
    features = [] 
    feats_zc = []
    feats_rms = []
    feats_AR = []
    data = random_schedule('random_training_data2022-08-04 10:15:23.292401.pkl',gest_map,num_trials,win_size,win_inc)
    for k in data.keys():
        feats_ges[k] = []
        for arr in (data[k]):
            feats = compute_features.features(win_size,win_inc,arr)
            zc = feats.zero_crossing()
            rms = feats.RMS()
            feats_zc.extend(zc)
            feats_rms.extend(rms)
            feats_ges[k].append(feats.zero_crossing())
            labels_zc.extend([gest_map[k]] * len(zc))
            labels_rms.extend([gest_map[k]] * len(rms))
        #print('len feats: ',k,':',len(feats_zc)+ len(feats_rms))
    #plot_EMG(feats_ges)
    features.extend(feats_zc)
    features.extend(feats_rms)
    features = np.array(features)
    #print('features shape: ',features.shape)
    labels.extend(labels_zc)
    labels.extend(labels_rms)
    labels = np.array(labels)
    clf = train_validate_model('KNN',features,labels)

    labels = []
    labels_zc = [] 
    labels_rms = []
    features = [] 
    feats_zc = []
    feats_rms = []
    feats_AR = []
    data2 = random_schedule('random_training_data2022-08-04 11:32:24.852911.pkl',gest_map,num_trials,win_size,win_inc)
    for k in data2.keys():
        for arr in (data2[k]):
            feats = compute_features.features(win_size,win_inc,arr)
            zc = feats.zero_crossing()
            rms = feats.RMS()
            feats_zc.extend(zc)
            feats_rms.extend(rms)
            labels_zc.extend([gest_map[k]] * len(zc))
            labels_rms.extend([gest_map[k]] * len(rms))
        print('len feats: ',k,':',len(feats_zc)+ len(feats_rms))
    #plot_EMG(feats_ges)
    features.extend(feats_zc)
    features.extend(feats_rms)
    features = np.array(features)
    #print('features shape: ',features.shape)
    labels.extend(labels_zc)
    labels.extend(labels_rms)
    labels = np.array(labels)
    X_test = features
    y_test = labels
    y_pred = clf.predict(X_test)
    
    print('test on dataset2: ',accuracy_score(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    print('CONF MAT')
    print(cf_matrix)
    #plot_EMG(data)
    #classifier = get_model(data,gest_map,win_size,win_inc,'KNN')
    #test_model('',classifier,gest_map,num_trials,win_size,win_inc)
    
