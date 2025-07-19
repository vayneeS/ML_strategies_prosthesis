from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import signal
import numpy as np
import math


class Classifier():
    def __init__(self, algo):
        self.algo = algo

    def train_validate(self, features, labels):
        features = np.array(features)
        labels = np.array(labels)
        print('features :', features.shape)
        print('labels :', labels.shape)

        skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)

        if (self.algo == "LDA"):
            model = LinearDiscriminantAnalysis(solver='svd')

        elif (self.algo == "KNN"):
            model = KNeighborsClassifier(
                n_neighbors=3, metric='minkowski', p=2)

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

    def iav(self, data):
        num_win = math.floor((data.shape[0] - self.win_size)/self.win_inc) + 1
        mn = np.zeros((num_win, data.shape[1]))
        start = 0
        end = self.win_size

        for i in range(num_win):
            cur_win = data[start:end, :]
            mn[i, :] = np.sum(np.absolute(cur_win), axis=0)
            start += self.win_inc
            end += self.win_inc
        return mn

    def zc(self, data):
        num_win = math.floor((data.shape[0] - self.win_size)/self.win_inc) + 1
        zc = np.zeros((num_win, data.shape[1]))
        start = 0
        end = self.win_size
        thresh = 1e-8

        for i in range(num_win):
            cur_win = data[start:end, :]
            for j in range(data.shape[1]):
                sc = (np.diff(np.sign(cur_win[:, j])) != 0)*1
                #ts = (np.absolute(np.diff(cur_win[:,j]))>thresh)*1
                # print(np.where(np.diff(np.signbit(cur_win[:,j]))))
                #zc[i,j] = len(np.where(np.diff(np.signbit(cur_win[:,j])))[0])
                zc[i, j] = np.sum(sc)
            start += self.win_inc
            end += self.win_inc
        print('zc:', zc)

        return zc


