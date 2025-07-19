from __future__ import print_function
import myo
import random
from time import perf_counter
import pickle
import datetime
import gesture_classifier
import numpy as np
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from numpy import unravel_index
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

class Emg(myo.DeviceListener):

  def __init__(self):
    super(Emg, self).__init__()
    # self.times = collections.deque()
    # self.times = []
    # self.last_time = None
    # self.n = int(n)
    self.emg = []

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    self.emg = event.emg

def get_data(gesture,num_samples):
    listener = Emg()
    samples = []
    myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')
     
    try:
        hub = myo.Hub('com.vaynee.myo-python')
        period_ms = 10
        print("\nGesture -- ", gesture, " : Ready?")
        input("Press Enter to continue...")
        
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
    except Exception as e:  
        print("Exception occurred for "+ repr(e))
    
    return samples

# def get_test_data(win_size,win_inc):
#     listener = Emg()
#     samples = []
#     emg_data_queue = deque(maxlen=win_size)
#     myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')
     
#     try:
#         hub = myo.Hub('com.vaynee.myo-python')
#         period_ms = 50
#         print("\n Ready?")
#         #input("Press Enter to continue...")
        
#         t1_start = perf_counter()
#         while hub.run(listener.on_event, period_ms):
#             if(len(listener.emg)>1):
#                 emg_data_queue.append(listener.emg)
#                 print(listener.emg)
#             if(len(emg_data_queue)>=win_size):
#                 samples.append(list(emg_data_queue))
#                 emg_data_queue.clear()
#                 features = get_features(np.array(samples),win_size,win_inc)
#         send_features(features)
  
#         #pickle_data(samples,'')
#         t1_stop = perf_counter()
#         print("Elapsed time:",  t1_stop-t1_start)
#         print(len(samples))
#     except:  
#         print("Unable to connect to Myo Connect. Is Myo Connect running?")
    
#     return 

def populate_gestures(num_trials=5):
    g_map = {'Wrist Pronation':0,'Wrist Supination':1, 'Hand Closing':2, 'Hand Opening':3, 'Pinch Closing':4, 'Pinch Opening':5,'Rest':6}
    #g_map = {'Wrist Pronation':0}
    gesture_set =[]
    for k in g_map.keys():
        gesture_set.extend([k] * num_trials)
    # for k in g_map.keys():
    #     gesture_set.append(k)
    g_rand = random.sample(gesture_set, len(gesture_set))
    return g_rand,g_map

def pickle_data(data,gesture):   
    """
    pickles emg data and gestures into file 
    """  
    curr_time = datetime.datetime.now()
    fname = gesture+str(curr_time)+'.pkl'
    #flog = 'gestures_log'+str(curr_time)+'.pkl'
    pickle.dump(data, open('data/testmodel2/'+fname, 'wb'))
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

#def get_features_labels(emg_data,map,gesture):
def compute_features(emg_data,win_size,win_inc):
    """returns num_windows x 8 features and num_windows labels
    """
    f = gesture_classifier.Features(win_size,win_inc)
    feat_rms = []
    # feat_iav = []
    # feat_zc = []
    rms = f.rms(emg_data)
    feat_rms.extend(rms) 
  
    # feat_iav.extend(f.iav(emg_data))
    # feat_zc.extend(f.zc(emg_data))
    #plot_emg(feat_zc)
    features = np.array(feat_rms)
    # features = np.concatenate((features, feat_iav, feat_zc), axis=1)
   
    return features


def train_model(algo,features,labels):
    model = gesture_classifier.Classifier(algo)
    model.train_validate(features,labels)
    return model

    
def plot_accuracy(dict):
    keys = list(dict.keys())
    acc=[float(dict[k][0]) for k in keys]
    algo = ['lda']
    std = [float(dict[k][1]) for k in keys]
    sns.barplot(x=keys, y=acc, hue = "")
    plt.show()

def pickle_features(num_samples,gesture,folder):
    dict_feats = {}

    data = get_data(gesture,num_samples)
    if not (gesture in dict_feats.keys()):
        dict_feats[gesture] = []
    features =compute_features(np.array(data),128,32)
    dict_feats[gesture].extend(features)
    
    print('shape feats: ',len(dict_feats[gesture]))
       
    curr_time = datetime.datetime.now()
    f="data/"+folder+"features"+str(curr_time)+".pkl"
    pickle.dump(dict_feats, open( f, "wb" ))

def compute_features(filename,map):
    data = open_pickle('data/'+filename)
    all_feats = []
    all_lbs = []
    for k in data.keys():
        feats = data[k]
        all_feats.extend(feats)
        all_lbs.extend([map[k]] * len(feats))
    all_feats= np.array(all_feats)
    all_lbs= np.array(all_lbs)
    return all_feats


if __name__ == '__main__':
#     all_features = []
#     all_labels = []
#     g_rand,g_map = populate_gestures(5)

#     #data_files = ['Hand Closing2022-09-13 17:11:40.364206.pkl','Hand Closing2022-09-13 17:14:08.872856.pkl','Hand Opening2022-09-13 17:11:21.953167.pkl','Hand Opening2022-09-13 17:13:50.220193.pkl','Pinch Closing2022-09-13 17:10:26.288719.pkl','Pinch Closing2022-09-13 17:13:11.915749.pkl','Pinch Opening2022-09-13 17:11:03.623754.pkl','Pinch Opening2022-09-13 17:12:36.054569.pkl','Rest2022-09-13 17:12:53.549053.pkl','Rest2022-09-13 17:14:26.963230.pkl','Wrist Pronation2022-09-13 17:10:44.741770.pkl','Wrist Pronation2022-09-13 17:13:30.825255.pkl','Wrist Supination2022-09-13 17:11:58.675522.pkl','Wrist Supination2022-09-13 17:12:17.171816.pkl']
#     # for f in data_files:
#         # data = open_pickle('data/'+f)
#         # print(data)
#         # str_ges = f[0:-30]
#         #features,labels = get_features_labels(np.array(data),g_map,str_ges)
#         # print('shape feats: ',features.shape)
#         # all_features.extend(features)
#         # all_labels.extend(labels)    
#     dict_feats = {}

#     for g in g_rand:
#         data = get_data(g,1000)
#         if not (g in dict_feats.keys()):
#             dict_feats[g] = []
#         features,labels = compute_features(np.array(data),128,32)
#         dict_feats[g].extend(features)
        
#         print('shape feats: ',features.shape)
#         #all_features.extend(features)
#         #all_labels.extend(labels)
#     curr_time = datetime.datetime.now()
#     f="data/training_features.pkl"+str(curr_time)
#     pickle.dump(dict_feats, open( f, "wb" ))   
#     #import os
#     # assign directory
#     # directory = 'data/trainmodel'
    
#     # # iterate over files in
#     # # that directory
#     # train_data_files= []
#     # for filename in os.listdir(directory):
#     #     f = os.path.join(directory, filename)
#     #     # checking if it is a file
#     #     if os.path.isfile(f):
#     #         train_data_files.append(str(f).split("trainmodel/",1)[1])
#     #print(train_data_files)
#     # train_data_files = ['Hand Closing2022-09-19 22:47:39.823724.pkl','Hand Opening2022-09-19 22:47:53.530805.pkl','Pinch Closing2022-09-19 22:47:10.577304.pkl','Pinch Opening2022-09-19 22:47:24.437338.pkl','Rest2022-09-19 22:47:16.834775.pkl','Wrist Pronation2022-09-19 22:47:32.844268.pkl','Wrist Supination2022-09-19 22:47:46.871961.pkl']
#     # for f in train_data_files:
#     #     train = open_pickle('data/trainmodel/'+f)
#     #     if not (f[0:-30] in dict_feats.keys()):
#     #         dict_feats[f[0:-30]] = []
#     #     features,labels = get_features_labels(np.array(train),g_map,f[0:-30],128,32)
#     #     dict_feats[f[0:-30]].extend(features)
#     #     #print('shape feats: ',features.shape)

#     #     all_features.extend(features)
#     #     all_labels.extend(labels)
#    #open pickled features and train
#     data_train = open_pickle('data/training_features')

#     modelKNN = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
#     modelSVM = svm.SVC(gamma=0.001)
#     modelLDA = LinearDiscriminantAnalysis(solver='svd')
#     models_arr = [modelKNN,modelSVM,modelLDA]
#     all_feats = []
#     all_lbs = []
#     for k in data_train.keys():
#         feats = data_train[k]
#         all_feats.extend(feats)
#         all_lbs.extend([g_map[k]] * len(feats))
#     all_feats= np.array(all_feats)
#     all_lbs= np.array(all_lbs)
#     # print('-----training---')
#     X_train, X_test, y_train, y_test = train_test_split(all_feats, all_lbs, test_size=0.33, random_state=42)
#     modelKNN.fit(X_train, y_train)
#     modelSVM.fit(X_train, y_train)
#     modelLDA.fit(X_train, y_train)

#     print('KNN: ',modelKNN.score(X_test, y_test))
#     print('SVM: ',modelSVM.score(X_test, y_test))
#     print('LDA: ',modelLDA.score(X_test, y_test))
#     #open test picked features 
#     all_features_test = []
#     all_labels_test = []
   
#     data_test = open_pickle('data/test_features')
  
#     for k in data_test.keys():
#         feats = data_test[k]
#         all_features_test.extend(feats)
#         all_labels_test.extend([k] * len(feats))
#     all_features_test= np.array(all_features_test)
#     all_labels_test= np.array(all_labels_test)

    data = open_pickle("data/features_2022-10-02 19:09:05.336086.pkl")
    print(data)
    # all_feats = []
    # for k in data.keys():
    #     feats = data[k]
    #     all_feats.extend(feats)
    # all_feats= np.array(all_feats)
    # data = np.array([[1,2,3,4,5,5,6,7],[3,5,8,0,6,4,8,5],[9,6,0,6,5,5,6,3],[1,2,7,4,5,5,6,7]])
    # def apply_threshold(data):
    #     data_trans = data.transpose()
    #     max_sig = np.max(data_trans)
    #     result = unravel_index(np.max(data_trans),data_trans.shape)
    #     max_index = result[0]
    #     print(max_index)
    #     #electrode = max_index[0][0]
    #     min_sig = np.min(data_trans[max_index])
    #     threshold = min_sig + (max_sig-min_sig)/4
    #     result = np.nonzero(data_trans[max_index] > threshold)[0]
    #     print(result)

    #     #tuple_indices = list(zip(result[0], result[1]))
    #     data_after_thresh = []

    #     for r in range(len(data_trans)):
    #         tmp = []
    #         for c in result:
    #             tmp.append(data_trans[r][c])

    #         data_after_thresh.append(tmp)
    #     data_after_thresh = np.array(data_after_thresh)

    #     return np.transpose(data_after_thresh)
    #         #data_trans[tuple_indices[ind][0]][tuple_indices[ind][1]]
    #         #data_after_thresh.append(data_trans[tuple_indices[ind]])
    #     # for row in range(len(data_trans)):
           
    #     #     print(threshold)
    #     #     for j in range(len(data_trans[row])):
    #     #         indices = (data_trans[row][j] > threshold)
    #     #         res.append(data_trans[row][j][indices])
    #     # return res
        

    # apply_threshold(data)



   
  
    

