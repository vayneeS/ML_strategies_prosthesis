import myo
import datetime
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
import json
import websockets
from time import perf_counter
import numpy as np
import math
from collections import deque
import logging
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import entropy
from sklearn.neural_network import MLPClassifier
import socket
import pickle
import statistics as st
import random
import os
from datetime import datetime

class Emg(myo.DeviceListener):

    def __init__(self):
        super(Emg, self).__init__()
        self.emg = []
        self.locked = False
        print("init myo")

    def on_connected(self, event):
        event.device.stream_emg(True)
        print("connected")

    def on_emg(self, event):
        self.emg = event.emg

    def on_paired(self, event):
        print("Paired, {}!".format(event.device_name))
        event.device.vibrate(myo.VibrationType.short)

    def on_unpaired(self, event):
        return False  # Stop the hub
    
    def on_locked(self, event):
        self.locked = False
        self.output()
    # def on_connect(self, myo, timestamp, firmware_version):
    #     print("connected")
    #     myo.set_stream_emg(myo.StreamEmg.enabled)
def separability(X, y):
    X_proj = X
    y = np.array(y)
    sep_ = {}
    class_sizes = {}
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        class_sizes[c] = len(idx_c)
        inter_var = []
        for c2 in np.unique(y):
            if c != c2:
               idx_c2 = np.where(y == c2)[0]
               X_ = np.r_[X_proj[idx_c], X_proj[idx_c2]]
               inter_var.append(np.mean(np.var(X_, axis=0)))
        mean_intra = np.mean(np.var(X_proj[idx_c], axis=0))
        print('  *', c, len(idx_c), '|', mean_intra, np.mean(inter_var), '|', np.var(X_proj[idx_c], axis=0), inter_var)
        if(mean_intra != 0):
            sep_[c] = np.mean(inter_var) / mean_intra
        else:
            print('mean_intra = 0')
            sep_[c] = np.mean(inter_var)
    # experimental ! 
    for c in class_sizes.keys():
        sep_[c] *= class_sizes[c] / np.sum([class_sizes[k] for k in class_sizes.keys()])
    return sep_

async def compute_entropy(X_train,y_train,ens_clf):    
    # clf = MLPClassifier(random_state=1, max_iter=300)
    # clf.fit(X_train, y_train)
    # ens_clf = [MLPClassifier(random_state=1, max_iter=300) for n in range(3)]
    for n in range(len(ens_clf)):
        ens_clf[n].fit(X_train, y_train)
    entropy_class = {}
    for k in range(len(X_train)):
        probas = []
        for n in range(len(ens_clf)):
            # v  = clf.predict_proba([X_train[k]])
            probas.append(
                ens_clf[n].predict_proba([X_train[k]])[0])
        v = np.mean(np.array(probas), axis=0)
        if y_train[k] not in entropy_class.keys():
            entropy_class[y_train[k]] = []
        entropy_class[y_train[k]].append(entropy(v))

    entropy_scores = {}
    for k in entropy_class.keys():
        entropy_scores[k] = np.mean(entropy_class[k])
    print('scores: ',entropy_scores)
    # with open('data/entropy.txt', 'a') as f:
    #     f.write(json.dumps(entropy_scores))
    return entropy_scores

# async def entropy_from_probs(labels,confidences):
#     #print('received data: ',confidences)
#     entropy_class = {}
#     for k in range(len(labels)):
#         if labels[k] not in entropy_class.keys():
#                 entropy_class[labels[k]] = []
        
#         entropy_class[labels[k]].append(entropy(np.array(list(confidences[k].values()))))
        
#     entropy_scores = {}
#     for k in entropy_class.keys():
#         entropy_scores[k] = np.mean(entropy_class[k])
#     print('scores: ',entropy_scores)
#     return entropy_scores

class Features():
    def __init__(self, win_size, win_inc):
        self.win_size = win_size
        self.win_inc = win_inc

    def rms(self, data):
        # print(data.shape)
        num_win = math.floor((data.shape[0] - self.win_size)/self.win_inc) + 1
        #print('num_win: ',num_win)
        rms = np.zeros((num_win, data.shape[1]))
        start = 0
        end = self.win_size

        for i in range(num_win):
            cur_win = data[start:end, :]
            rms[i, :] = np.sqrt((np.square(cur_win)).mean(axis=0))
            start += self.win_inc
            end += self.win_inc
        return rms


def compute_features(emg_data, win_size, win_inc):
    """returns num_windows x 8 features 
    """
    f = Features(win_size, win_inc)
    feat_rms = []

    rms = f.rms(emg_data)
    feat_rms.extend(rms)

    features = np.array(feat_rms)

    return features

# async def get_EMG_features(websocket,win_size, win_inc):
#     print("get data")
#     samples = []
#     listener = Emg()
#     features = np.array([])
#     period_ms = 10
#     emg_data_queue = deque(maxlen=win_size)
#     test_collect = []

#     hub = myo.Hub('com.niklasrosenstein.myo-python')
#     t1_start = perf_counter()
#     try:
#         #update our display once every 10ms, 1s=100 updates
#         while hub.run(listener.on_event, period_ms):
                
#             if(len(listener.emg)>1):
#                 emg_data_queue.append(listener.emg)

#             if(len(emg_data_queue)>=win_size):
#                 #print('emg data size: ',len(emg_data_queue))

#                 samples = list(emg_data_queue)
#                 try:
#                     features = compute_features(np.array(samples), win_size, win_inc)
#                     #print(features)
#                     t1_stop = perf_counter()
#                     elapsed_time = t1_stop-t1_start    

#                     await send_features(websocket,features,"features")
#                     test_collect.append(features)
#                     if(len(test_collect) > 2):
#                         print('test collect length',len(test_collect),elapsed_time)
#                         break
#                 except Exception as e:
#                     print("Exception occurred for " + repr(e))  
#                 #return features         
#     except Exception as e:
#         print("Exception occurred for " + repr(e))
        
def get_EMG_features(win_size, win_inc, listener, hub):
    ######################################
    #perf_counter()-returns the float value of time in seconds
    ######################################
    print("get data")
    #samples = []
    features = np.array([])
    period_ms = 10
    #emg_data_queue = deque(maxlen=win_size)
    buffer_samples = []
    # avg_features = []
    t1_start = perf_counter()
    # tmp_counter = 0
    try:
        #update our display once every 10ms, 1s=100 updates
        while hub.run(listener.on_event, period_ms):
                
            # if(len(listener.emg)>1):
            #     emg_data_queue.append(listener.emg)
            #     buffer_samples.append(listener.emg)

            # if(len(emg_data_queue)==win_size):
                # print('emg data size: ',len(emg_data_queue))
                # if tmp_counter < 2:
                #     print(emg_data_queue)
                #     tmp_counter += 1
                # samples = list(emg_data_queue)
            try:
                if(len(listener.emg)>1):
                    # emg_data_queue.append(listener.emg)
                    buffer_samples.append(listener.emg)

                    # buffer_samples.append(list(emg_data_queue))
                    t1_stop = perf_counter()
                    elapsed_time_sec = t1_stop - t1_start
                    # avg_samples = np.mean(np.array(buffer_samples),axis = 0)
                    # await send_features(websocket,features,"features")
                    # print('buffer: ',np.array(buffer_samples).shape)
                    
                    if(elapsed_time_sec > 2):
                        features = compute_features(
                            np.array(buffer_samples), 
                            win_size, win_inc)
                        print('buffer_samples shape: ', np.array(buffer_samples).shape)
                        print('features shape: ', features.shape)                        
                        if(len(features) > 0):
                            features = np.mean(features, axis=0).reshape(1,-1)
                            print('avg features shape: ', features.shape)
                            break
                        #segfault
            except Exception as e:
                print("Exception occurred for " + repr(e))  
            #return features         
    except Exception as e:
        print("Exception occurred for " + repr(e))
    finally:
        hub.stop()     
    return features

from sklearn.exceptions import NotFittedError

async def respond_handler(websocket, clf, win_size=128, win_inc=32):
    print("respond_handler")
    features = []
    listener = Emg()
    hub = myo.Hub('com.niklasrosenstein.myo-python')
    while True:  # keep connection open
        try:
            message = await websocket.recv()
            message = json.loads(message)
            if(message == 'start'):
                print('start')
                # await get_EMG_features(websocket,win_size, win_inc)
                features = get_EMG_features(win_size, win_inc,listener,hub)
                await send_features(websocket, features, 'features')           
                #print(features)
            if(not isinstance(message,str)):
                if('action' in message.keys()):
                    if(message['action'] == 'cmd'):
                        print('received cmd')
                        s.send(int.to_bytes(message['content'], 1, "big"))
                    if(message['action'] == 'prediction'):
                        # y_preds = []
                        # y_mode = []
                        features=features.reshape(1,-1)
                        # features = features.tolist()
                        # print(x.shape)
                
                        try:  
                            # for clf in ens_clf:
                            #     y_preds.extend(clf.predict(features))
                            # y_mode=st.mode(y_preds)
                            # print('labels: ',y_mode)
                            y_pred = clf.predict(features).tolist()
                            print(y_pred)
                            await websocket.send(json.dumps({"prediction": y_pred[0]}))
                        except Exception as e:
                            print(repr(e))
                
                try:
                    if(len(message.keys())>0):
                        if("X_train" in message.keys()):
                            X_train,y_train = message['X_train'],message['y_train']
                            # if(message["action"] == 'entropy'):
                            #     entropy_scores = await compute_entropy(X_train,y_train,ens_clf)
                            #     gesture = max(entropy_scores,key=entropy_scores.get)
                            #     print('gesture entropy: ',gesture)
                            #     if(gesture != None):               
                            #         await websocket.send(json.dumps({"gesture": gesture}))
                            if(message['action'] == 'train'):
                                # for n in range(len(ens_clf)):
                                #     ens_clf[n].fit(X_train, y_train)
                                phase = message['phase'] 
                                trial = message['trial']
                                clf.fit(X_train, y_train)
                                curr_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                filename = './data/models/model_'+phase+'_'+str(trial)+'_'+curr_time+'.pickle'
                                pickle.dump(clf, open(filename, 'wb'))
                                print('trained')
                            if(message['action'] == 'sep'):
                                print('callback sep')
                                # clf.fit(X_train, y_train)#to remove
                                trial = message['trial']
                                try:
                                    sep = separability(clf.transform(X_train), y_train)
                                    print('separability', sep)
                                    filename = './data/separability/sep_'+str(trial)+'_'+curr_time+'.pickle'
                                    pickle.dump(sep, open(filename, 'wb'))
                                    sorted_sep = sorted(sep.items(), key=lambda x:x[1])

                                    gesture = sorted_sep[0][0]
                                    print(gesture)
                                    await websocket.send(json.dumps({"sep_gesture": gesture}))
                                except NotFittedError as e:
                                    print(repr(e))
                                    clf.fit(X_train, y_train)
                                    sep = separability(clf.transform(X_train), y_train)
                                    print('separability', sep)
                                    filename = './data/separability/sep_'+str(trial)+'_'+curr_time+'.pickle'
                                    pickle.dump(sep, open(filename, 'wb'))
                                    sorted_sep = sorted(sep.items(), key=lambda x:x[1])

                                    gesture = sorted_sep[0][0]
                                    print(gesture)
                                    await websocket.send(json.dumps({"sep_gesture": gesture}))
                            
                except Exception as e:
                    print(repr(e))        
                    
        except Exception as e:
            print(repr(e))
        # except websockets.ConnectionClosedOK:
        #     break
        except websockets.ConnectionClosed:
            print(f"Terminated")
            break

    
async def send_features(websocket, features, msg_type):
    print(features)
    # time_feats = {}
    features_lst = features.tolist()
    # time_feats[elapsed_time] = feats
    #print("len feats:", len(features_lst))
    await websocket.send(json.dumps({"type": msg_type, "features": features_lst}))

#manages a websocket connection : When handler terminates, websockets closes the connection
async def handler(websocket):
    myo.init(sdk_path='../../sdk/')
    # ens_clf = [MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1) for n in range(3)]
    clf = LinearDiscriminantAnalysis()
    background_tasks = set()
    print("handler")
    try:
        task_respond = asyncio.create_task(respond_handler(websocket,clf))
        await task_respond
        background_tasks.add(task_respond)
        task_respond.add_done_callback(background_tasks.discard)
    except asyncio.TimeoutError:
        print('task canceled')
    except Exception as e:
        print(repr(e))
        await websocket.close()

if __name__ == "__main__":
    ### server###
    loop = asyncio.get_event_loop()
    #loop.set_debug(True)

    print("...Server Ready!")
    try:        
        logging.basicConfig(level=logging.DEBUG)
        #start a websockets server
        start_server = websockets.serve(handler, host="localhost", port=8765)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.0.0.1',45458))
        loop.run_until_complete(start_server)
        loop.run_forever()
        #asyncio.get_event_loop.close()
    
    except Exception as e:
        print(repr(e))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()