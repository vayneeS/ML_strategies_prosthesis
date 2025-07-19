from threading import Lock
from collections import deque
import time
import random 
import myo

class Listener(myo.DeviceListener):
    def __init__(self, queue_size=1):
        #self.n = n
        self.lock = Lock()
        self.ori_data_queue = deque(maxlen=queue_size)
        self.emg_data_queue = deque(maxlen=queue_size)

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
            self.emg_data_queue.append(event.emg)
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

    def set_sdk(self):
        myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')

    def start_hub(self):
        hub = myo.Hub()
        return hub
# if __name__ == '__main__':
#     myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')
#     hub = myo.Hub()
#     listener = Listener()
    
#     g_map = {'Wrist Pronation':0,'Wrist Supination':1, 'Hand Closing':2, 'Hand Opening':3, 'Pinch Closing':4, 'Pinch Opening':5,'Rest':6}
#     g_rand = random.sample(g_map.keys(), len(g_map.keys()))
#     emg_data = {}
#     for g in g_rand:
#         emg_data[g] = []
#         print("\nGesture -- ", g, " : Ready?")
#         input("Press Enter to continue...")
#         start_time = time.time()

#         while hub.run(listener.on_event, duration_ms=50):
#             emg_data_tmp = listener.on_update_emg()
#             for emg in emg_data_tmp:
#                 emg_data[g].append(emg)
#             #print(emg_data)
#             if time.time() - start_time > 10:
#                 break
#         print('num samples: ',len(emg_data[g]))