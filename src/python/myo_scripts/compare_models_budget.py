import random
from time import perf_counter
import pickle
import datetime
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import gesture_classifier
import sys   
import myo
import os


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

def open_pickle(path):

    with (open(path, "rb")) as f:
        while True:
            try:
                data = (pickle.load(f))
            except EOFError:
                break
    return data

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

def record_data(g_map,hub):
    keys = g_map.keys()
    dict_feats = {}
    curr_time = datetime.datetime.now()

    for k in keys:
        if not (k in dict_feats.keys()):
    #k='Pinch Opening'
            dict_feats[k] = []
            for i in range(42):
                try:
                    data = get_data(k,300,10,hub)
                    features = compute_features(np.array(data),128,32)
                    dict_feats[k].extend(features)
                except Exception as e:
                    print('Interrupted',repr(e))
                    try:
                        f="data/"+"features_"+str(curr_time)+".pkl"
                        pickle.dump(dict_feats, open( f, "wb" ))
                        sys.exit(0)
                    except SystemExit:
                        os._exit(0)
        f="data/"+"features_"+k+str(curr_time)+".pkl"
        pickle.dump(dict_feats, open( f, "wb" ))

def get_data(gesture,num_samples,period,hub):
    listener = Emg()
    samples = [] 
    try:
        print("\nGesture -- ", gesture, " : Ready?")
        input("Press Enter to continue...")
        
        t1_start = perf_counter()
        #if(not hub.running):
        while hub.run(listener.on_event, period):
            if(len(listener.emg)>1):
                samples.append(listener.emg)
            if(len(samples)==num_samples):
                break
        #pickle_data(samples,gesture)
        t1_stop = perf_counter()
        print("Elapsed time:",  t1_stop-t1_start)
        print("len samples:",len(samples))
    except Exception as e:  
        print("Exception occurred for "+ repr(e))
    #hub.shutdown()    
    return samples

if __name__ == '__main__':

    myo.init(sdk_path='../../../myo-sdk/')
    
    #period_ms = 10
    g_map = {'Wrist Pronation':0,'Wrist Supination':1,'Hand Closing':2, 'Hand Opening':3, 'Pinch Closing':4, 'Pinch Opening':5,'Rest':6,'Index Point':7}
    g_lbs = ['WP','WS','HC','HO','PC','PO','R','IP']
    hub = myo.Hub('com.niklasrosenstein.myo-python')

    record_data(g_map,hub)

    # modelKNN = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
    # modelSVM = svm.SVC(gamma=0.001)
    # modelLDA = LinearDiscriminantAnalysis(solver='svd')
    # models_arr = [modelKNN,modelSVM,modelLDA]
    # model_name = ['KNN','SVM','LDA']
  
    
    # # print('-----training---')
 
    # budget = [2,3,5,10,15,20,30,50,100,150,300]
    # #2,3,5,10,15,20,30,50,100,150,300]
 
    # data_test = open_pickle('data/test_features')
    # data_train = open_pickle('data/training_features')

    # result = {}
   
    # result['budget'] = []
    # result['score'] = []
    # result['model'] = []
    # result['algo'] = []
   
  
    # # print('----test-----')

    # feats_test = []
    # lbls_test = []
    # for k in data_test.keys():
    #     feats = data_test[k]
    #     feats_test.extend(feats)
    #     lbls_test.extend([g_map[k]] * len(feats))

    # batch_feats = []
    # batch_lbls = []
    # c = -1
    # # fig, axs = plt.subplots(len(model_name),len(budget), figsize=(20, 15), sharey=True)
    # avg_conf_mat_mdl = {}
    # for b in budget:
    #     c += 1
    #     for j in range(len(models_arr)):
    #             avg_conf_mat_mdl[j] = []
    #     for i in range(20):
    #         batch_feats.clear()
    #         batch_lbls.clear()
    #         for k in data_train.keys():
    #             batch = []
    #             feats = np.array(data_train[k])
    #             for l in range(b):
    #                 idx = random.randint(0,len(feats)-1)
    #                 while(idx in batch):
    #                     idx = random.randint(0,len(feats)-1)
    #                 batch.append(idx)
    #             batch_feats.extend(feats[batch])
    #             batch_lbls.extend([g_map[k]] * b)
    #         #print(len(batch_feats),len(batch_lbls))
    #         for j in range(len(models_arr)):           
    #             result['budget'].append(b)
    #             mdl = models_arr[j].fit(batch_feats,batch_lbls)
    #             result['score'].append(mdl.score(feats_test,lbls_test))
    #             result['model'].append(mdl)
    #             result['algo'].append(model_name[j])
    #             pred = mdl.predict(feats_test)
    #             conf_mat = confusion_matrix(lbls_test, pred)
    #             avg_conf_mat_mdl[j].append(conf_mat)
    #     # for j in range(len(models_arr)):
    #     #     sum_per_mdl = sum(avg_conf_mat_mdl[j])
    #     #     print(sum_per_mdl/20)
    #     #     disp = ConfusionMatrixDisplay((sum_per_mdl/20).astype(int),
    #     #     display_labels=g_lbs)
    #     #     disp.plot(ax=axs[j,c])
    #     #     disp.ax_.set_title(model_name[j] +'_'+ str(b),fontweight="bold")


    # #plt.show()

    # algo = ['KNN','SVM','LDA']
    
    # dataset =pd.DataFrame(result)

    # #print(dataset) 

    # sns.barplot(x='budget', y='score', data=dataset, hue='algo')
    
    # plt.show()
