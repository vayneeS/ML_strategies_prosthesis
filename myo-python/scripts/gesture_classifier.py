from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import math

class Classifier():
    def __init__(self,algo ):
        self.algo = algo

    def train_validate(self,features,labels):
        print('features :' ,features.shape)
        print('labels :' ,labels.shape)

        skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)

        if(self.algo == "LDA"):
            model = LinearDiscriminantAnalysis(solver='svd')
            
        elif(self.algo == "KNN"):
            model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)

        accuracies = []
        
        for train_index, test_index in skf.split(features, labels):

            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
        
            accuracies.append(accuracy_score(y_test, y_pred))
        print(np.mean(accuracies), np.std(accuracies))
        return model

class Features():
    def __init__(self):
        self.win_size = 128
        self.win_inc = 32
      
    def rms(self,data):
        num_win = math.floor((data.shape[0] - self.win_size)/self.win_inc) + 1        
        rms = np.zeros((num_win, data.shape[1]))
        start = 0
        end = self.win_size

        for i in range(num_win):
            cur_win = data[start:end, :]
            rms[i, :] = np.sqrt((np.square(cur_win)).mean(axis=0))
            start += self.win_inc
            end += self.win_inc
        return rms

    def zc(self,data):
        num_win = math.floor((data.shape[0] - self.win_size)/self.win_inc) + 1        
        zc = np.zeros((num_win, data.shape[1]))
        start = 0
        end = self.win_size
        thresh = 1e-8

        for i in range(num_win):
            cur_win = data[start:end, :]
            for j in range(data.shape[1]):
                sc = (np.diff(np.sign(cur_win[:,j])) != 0)*1
                ts = (np.absolute(np.diff(cur_win[:,j]))>thresh)*1
                #print(np.where(np.diff(np.signbit(cur_win[:,j]))))
                #zc[i,j] = len(np.where(np.diff(np.signbit(cur_win[:,j])))[0])
                zc[i,j] = np.sum(sc&ts)
                #print('zc: ',zc[i,j], 'data: ', cur_win[:,j].shape)
            start += self.win_inc
            end += self.win_inc
        return zc
    
    def ar(self,data,order):
        num_win = math.floor((data.shape[0] - self.win_size)/self.win_inc) + 1
        ar = np.zeros((num_win, data.shape[1]*order))
        start = 0
        end = self.win_size
        
        for i in range(num_win):
            cur_win = data[start:end, :]
            mod_coeff = []
            for j in range(data.shape[1]):
                mod = AutoReg(cur_win[:,j], lags=order).fit()
                mod_coeff.extend(mod.params[1:])
                print(len(mod_coeff))
            ar[i,:] = np.array(mod_coeff)
            start += self.win_inc
            end += self.win_inc
        print(ar.shape)
        return ar


if __name__ == '__main__':
    f = Features()
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_pacf
    import pickle
    with (open('emg_data2022-08-23 14:59:24.721015.pkl', "rb")) as f:
        while True:
            try:
                data = (pickle.load(f))
            except EOFError:
                break
    for k in data.keys():
        d = data[k]
        d_ = np.reshape(d,(-1,8))
    
    result = adfuller(d_[:,7])
    print('p-value: %.2f' % result[1])
    import matplotlib.pylab as plt

    plot_pacf(d_[:,0], lags=40)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Partial Autocorrelation', fontsize=12)
    plt.title('Partial Autocorrelation of First Order Differenced Series', fontsize=14)
    #plt.show()

    from statsmodels.tsa.arima_model import ARIMA
    import statsmodels.api as sm
    
    #mod = ARIMA(d_[:,0][:int(len(d_[:,0]))], order=(4, 0, 0))
    
    ar_model = AutoReg(d_[:,0], lags=4).fit()
    print(ar_model.params)