import math
import numpy as np

class features():
    def __init__(self, win_size, win_inc, data):
        self.win_size = win_size
        self.win_inc = win_inc
        self.data = data
        
    def RMS(self):   
        #print('shape rms: ',self.data.shape)
        num_win = math.floor((self.data.shape[0] - self.win_size)/self.win_inc) + 1        
        rms = np.zeros((num_win, self.data.shape[1]))
        start = 0
        end = self.win_size

        for i in range(num_win):
            cur_win = self.data[start:end, :]
            rms[i, :] = np.sqrt((np.square(cur_win)).mean(axis=0))
            start += self.win_inc
            end += self.win_inc
        return rms

    def zero_crossing(self):
        #print('shape zc: ',self.data.shape)
        num_win = math.floor((self.data.shape[0] - self.win_size)/self.win_inc) + 1        
        zc = np.zeros((num_win, self.data.shape[1]))
        start = 0
        end = self.win_size

        for i in range(num_win):
            cur_win = self.data[start:end, :]
            for j in range(self.data.shape[1]):
                sc = (np.diff(np.sign(cur_win[:,j])) != 0)*1
                max_sig = np.max(cur_win[:,j])
                min_sig = np.min(cur_win[:,j])
                threshold = (abs(max_sig)+abs(min_sig))/2
                ts = (np.absolute(np.diff(cur_win[:,j]))>threshold)*1
                zc[:,j] = np.sum(sc & ts)
            start += self.win_inc
            end += self.win_inc
        return zc
    
    def AR(self,order):
        
        return 
