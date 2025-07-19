from threading import Lock
from collections import deque
import myo
import numpy as np

class Listener(myo.DeviceListener):
    def __init__(self,queue_size=1):
        #self.n = n
        self.lock = Lock()
        #self.ori_data_queue = deque(maxlen=queue_size)
        #self.emg_data_queue = deque(maxlen=queue_size)
        #self.emg_data_queue = deque(maxlen=n)
        self.emg_data = []

    def get_emg_data(self):
        with self.lock:
            return self.emg_data

    def on_connected(self, event):
        event.device.stream_emg(True)
        self.emg_enabled = True

    # def on_connect(self, timestamp, firmware_version):
     #   myo.set_stream_emg(myo.StreamEmg.enabled)

    def on_emg(self, event):
        with self.lock:
            self.emg_data=event.emg
            #self.emg_data_queue.append((event.timestamp, event.emg))
            
            
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

    def clear_emg(self):
        self.emg_data.clear()

    def set_sdk(self):
        #myo.init(sdk_path='../../myo-sdk/')
        myo.init(sdk_path='/Users/vaynee_sungeelee/Desktop/mab_classifier/myo-sdk/')

    def get_hub(self):
        hub = myo.Hub()
        return hub
