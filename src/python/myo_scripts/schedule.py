import random
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import logging
# import daiquiri
# import daiquiri.formatter
# logger = daiquiri.getLogger(__name__)
import matplotlib.pylab as plt


class Schedule():

    def __init__(self):
        self.g_map = {'Wrist Pronation': 0, 'Wrist Supination': 1, 'Hand Closing': 2,
                      'Hand Opening': 3, 'Pinch Closing': 4, 'Pinch Opening': 5, 'Rest': 6, 'Index Point': 7}

    def get_random(self, num_trials):
        gesture_set = []
        for k in self.g_map.keys():
            gesture_set.extend([k] * num_trials)
        random_schedule = random.sample(gesture_set, len(gesture_set))
        return random_schedule
    
    async def compute_separability_index(self, data):
        #print("predictions:",predictions)
        #print(dict_classification)
        separability_indices = {}
        variances = {}
        amplitudes = {}
        
        clf = LinearDiscriminantAnalysis() #n_components=2)
        X = []
        y = []
        for cls in data.keys():
            for x in data[cls]:
                X.append(x)
                y.append(cls)
        X = np.array(X)
        y = np.array(y)
        print(X.shape, len(y))
        
        data_trsf = clf.fit_transform(X, y)
        data_lda = {}
        for ci, cls in enumerate(y):
            if cls not in data_lda.keys():
                data_lda[cls] = []
            data_lda[cls].append(data_trsf[ci])

        # n_classes = np.arange(8)
        # target_names = np.unique(y)
        # plt.figure(figsize=(12,12))
        # for i, target_name in zip(n_classes, target_names):
        #     print(y, target_name, y == target_name)
        #     print(data_trsf[y == target_name, 1])
        #     plt.scatter(data_trsf[y == target_name, 0], data_trsf[y == target_name, 1], label=target_name)
        # plt.legend()
        # plt.savefig('test_lda.png')
        
        
        for key in data.keys():
            intra_class_var = np.mean(np.var(data[key], axis=0))
            intra_class_var_lda = np.mean(np.var(data_lda[key], axis=0))
            variances[key] = {}
            variances[key]['intra'] = intra_class_var_lda #intra_class_var
            variances[key]['inter'] = 0
            amplitudes[key] = np.sqrt(np.sum(np.power(data_lda[key], 2)))
            sep_ = []
            for key2 in data.keys():
                if key != key2:
                    inter_class_var = np.mean(np.var(np.r_[data[key], data[key2]], axis=0))
                    inter_class_var_lda = np.mean(np.var(np.r_[data_lda[key], data_lda[key2]], axis=0))
                    if(intra_class_var != 0):
                        #sep_.append(inter_class_var / intra_class_var)
                        sep_.append(inter_class_var_lda / intra_class_var_lda)
                    # else:
                    #     sep_.append(inter_class_var)
                    variances[key]['inter'] += inter_class_var_lda
            separability_indices[key] = np.mean(sep_)        
            print("separability indices:", key, separability_indices[key], variances[key], amplitudes[key])
       
        return separability_indices

    async def draw_exp3(self, separability_indices, experiment):
        #experiment = MAB.EXP3(len(self.g_map))
        #experiment.gamma = 0.07

        reward_vect = []
        for ind in list(separability_indices.values()):
            reward_vect.append(1/ind)
        print("rewards:", reward_vect)
        choice, reward, est, weights = experiment.exp3(reward_vect)
        #bestAction = max(range(experiment.numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))
        print('choice: ', choice, ',reward: ', reward,
              ',estimation: ', est, ',weights: ', weights)
        return list(separability_indices.keys())[choice]

