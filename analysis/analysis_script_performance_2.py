import os
import csv
import json
import pickle
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

def load_data(data_path):
    '''
    Load data from a folder organised by COND and PID with .db files and models/ folder
    Take the folder name to browse as arguments
    Return the data and the models 
    '''
    res = []
    models = {}
    data = {}
    # Iterate directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # handle db files (movement data basically)
            if file.endswith(".db"):
                # participant id 
                pid = int(root.split('/')[2][1:])
                # curriculum
                cond = root.split('/')[1]
                # data structure 
                if(pid not in data.keys()):
                    data[pid] = {}
                    data[pid]['cond'] = cond
                    data[pid]['training'] = {}
                    data[pid]['training']['x'] = []
                    data[pid]['training']['y'] = []
                    data[pid]['posttest'] = {}
                    data[pid]['posttest']['x'] = []
                    data[pid]['posttest']['y'] = []
                    data[pid]['negative'] = {}
                    data[pid]['negative']['x'] = []
                    data[pid]['negative']['y'] = []
                    data[pid]['positive'] = {}
                    data[pid]['positive']['x'] = []
                    data[pid]['positive']['y'] = []
                with open(os.path.join(root, file)) as f:  
                    content = f.read().splitlines()
                    if('training' in file):
                        times = {}
                        for j in range(len(content)):
                            content_json = json.loads(content[j])
                            times[content_json['createdAt']['$$date']] = j
                        for k in sorted(times.keys()):
                            content_json = json.loads(content[times[k]])
                            y = content_json['y']
                            x = content_json['x']
                            data[pid]['training']['x'].append(x)
                            data[pid]['training']['y'].append(y)
                    elif('posttest' in file):
                        for j in range(len(content)):
                            content_json = json.loads(content[j])
                            y = content_json['y']
                            x = content_json['x']
                            data[pid]['posttest']['x'].append(x)
                            data[pid]['posttest']['y'].append(y)
                    elif('pos-neg' in file):
                        for j in range(len(content)):
                            content_json = json.loads(content[j])
                            y = content_json['y']
                            x = content_json['x']
                            # data[pid]['posttest']['x'].append(x)
                            # data[pid]['posttest']['y'].append(y)
                            if 'Negative' in y:
                                ny = ' '.join(y.split(' ')[:-1])
                                data[pid]['negative']['x'].append(x)
                                data[pid]['negative']['y'].append(ny)
                            elif 'Positive' in y:
                                ny = ' '.join(y.split(' ')[:-1])
                                data[pid]['positive']['x'].append(x)
                                data[pid]['positive']['y'].append(ny)

            # handle pickle files (models)
            if file.endswith(".pickle"):
                # participant id 
                pid = int(root.split('/')[2][1:])
                if('model' in file):                     
                    phase_id = int(file.split('_')[1][-1])
                    if phase_id == 2:
                        if pid not in models.keys():
                            models[pid] = {}
                            models[pid]['training'] = {}
                        trial_id = int(file.split('_')[2])
                        models[pid]['training'][trial_id] = pickle.load(open(os.path.join(root, file), 'rb'))
                    elif phase_id == 1:
                        if pid not in models.keys():
                            models[pid] = {}
                            models[pid]['init'] = {}
                        if 'init' not in models[pid].keys():
                            models[pid]['init'] = {}
                        models[pid]['init'][1] = pickle.load(open(os.path.join(root, file), 'rb'))
    # check outlier baased on training accuracy
    data_sns = {'pid': [], 'curriculum': [], 'accuracy': []}
    outliers = []
    for pid in data.keys():
        classifier = LinearDiscriminantAnalysis()
        X = np.array(data[pid]['training']['x'])
        y = np.array(data[pid]['training']['y'])
        classifier.fit(X, y)
        data_sns['pid'].append(pid)
        data_sns['curriculum'].append(data[pid]['cond'])
        data_sns['accuracy'].append(
        classifier.score(
            np.array(data[pid]['posttest']['x']),
            np.array(data[pid]['posttest']['y'])))
    stds = {}
    meas = {}
    for c in np.unique(data_sns['curriculum']):
        idx = np.where(np.array(data_sns['curriculum']) == c)[0]
        acs = [data_sns['accuracy'][i] for i in idx]
        stds[c] = np.std(acs)
        meas[c] = np.mean(acs)
        for i in idx:
            if data_sns['accuracy'][i] <= meas[c] - 3*stds[c]:
                print('Outlier detected', c, data_sns['pid'][i], data_sns['accuracy'][i], meas[c], stds[c])
                outliers.append(data_sns['pid'][i])
    data2 = {}
    for pid in data.keys():
        if pid not in outliers:
            data2[pid] = data[pid]
    return data2, models

def perclass_accuracy(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes).T
    perclass_accuracy_ = {}
    for r in range(len(cm)):
        total = 0
        for c in range(len(cm)):
            if r == c:
                diag = cm[r,c]
            total += cm[r,c]
        #print(r, diag, total)
        if total != 0:
            perclass_accuracy_[classes[r]] = diag / total
        else:
            perclass_accuracy_[classes[r]] = 1.0
    return perclass_accuracy_

def perclass_recall(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes).T
    print(cm)
    perclass_recall_ = {}
    tp = []
    fn = []
    for c in range(len(cm)):
        total = 0
        for r in range(len(cm)):
            if(r==c): 
                tp = cm[r,c]
            total += cm[r,c]
            
        print(c, tp, total)
        if total != 0:
            perclass_recall_[c] = tp / total
        else:
            perclass_recall_[c] = 1.0
    return perclass_recall_

def questionnaires(file,data,models,model_id=120):
    #----------------------------------------------------------------#
    #   populate accuracy from questionnaire percentages, index of responses starts at 6
    #   file: file path
    #   data: training and test data in the format data[pid][str][str]
    #   models: model files indexable as models[pid]['training'][model_id]
    #----------------------------------------------------------------#
    df = pd.read_csv(file,delimiter=',') 
    cols= {'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Palm Up]' : 'Palm Up',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Palm Down]' : 'Palm Down',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Rest Hand]' : 'Rest Hand',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Open Hand]' : 'Open Hand',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Close Hand]' : 'Close Hand',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Open Pinch]' : 'Open Pinch',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Close Pinch ]' : 'Close Pinch',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Point Index]' : 'Point Index'}
    df.rename(columns=cols, inplace=True)
    df_dict = df.to_dict()
    #print(df_dict)
    participant_answer = {'UCC':{},'RC':{},'GSC':{}}
    # acc = compute_per_class_accuracy_training(data,models,model_id)
    # print(acc)
    
    gesture_names = []
    new_cond = {'UC':'UCC','NUC':'RC','NUCS':'GSC'}
    data_sns = {'pid':[],'curriculum':[],'perceived':[],'actual':[],'class':[]}
    d_temp = {'UCC':{},'RC':{},'GSC':{}}
    for i in range(len(df_dict.keys())):
        if(i >= 6 and i < 14):
            gesture_names.append(list(df_dict.keys())[i])
    # print(gesture_names)
    for index in df_dict['PID'].keys():
        pid = df_dict['PID'][index]
        if(pid in data.keys()):
            # print(pid)
            cond =  new_cond[df_dict['curriculum'][index]]
            classifier = models[pid]['training'][model_id] 
            accuracy = perclass_accuracy(data[pid]['training']['y'],
                                    classifier.predict(np.array(data[pid]['training']['x'])),
                                    classifier.classes_)
            #print(accuracy)
            d_temp[cond][pid] = accuracy
            temp_list = [d_temp[cond][pid][name] for name in sorted(d_temp[cond][pid].keys())]
            temp_list = stats.zscore(temp_list)
            for i, name in enumerate(sorted(d_temp[cond][pid].keys())):
                d_temp[cond][pid][name] = temp_list[i]

            if((pid not in participant_answer[cond].keys()) #and (pid in accuracy[cond].keys())
            ):
                participant_answer[cond][pid] = {} 
                for name in gesture_names:
                    if(name not in participant_answer[cond][pid].keys()):
                        participant_answer[cond][pid][name] = int(df_dict[name][index][0:-1])/100
                temp_list = [participant_answer[cond][pid][name] for name in sorted(participant_answer[cond][pid].keys())]
                temp_list = stats.zscore(temp_list)
                for i, name in enumerate(sorted(participant_answer[cond][pid].keys())):
                    participant_answer[cond][pid][name] = temp_list[i]
       
    correlations = {'UCC':[],'RC':[],'GSC':[]}
    for curriculum in participant_answer.keys():
        for pid in participant_answer[curriculum].keys():  
            answer = []
            model_acc = []
            for gesture in participant_answer[curriculum][pid].keys():
                #print(gesture)
                # answer.append(int(participant_answer[curriculum][pid][gesture][0:-1])/100)
                # answer.append(participant_answer[curriculum][pid][gesture])
                # model_acc.append(accuracy[curriculum][pid][gesture])
                # model_acc.append(d_temp[curriculum][pid][gesture])
                data_sns['pid'].append(pid)
                data_sns['curriculum'].append(curriculum)
                # data_sns['perceived'].append(int(participant_answer[curriculum][pid][gesture][0:-1])/100)
                data_sns['perceived'].append(participant_answer[curriculum][pid][gesture])
                data_sns['actual'].append(d_temp[curriculum][pid][gesture])
                data_sns['class'].append(gesture)
            #print(pid, answer)
            #sorted_answer_indices = np.argsort(np.array(answer))
            # selected_answer = np.concatenate((np.array(answer)[sorted_answer_indices[:2]],np.array(answer)[sorted_answer_indices[-2:]]))
            # selected_model_acc = np.concatenate((np.array(model_acc)[sorted_answer_indices[:2]],np.array(model_acc)[sorted_answer_indices[-2:]]))
            # if(pid == 22):
            #     print(min_max_answer)
            #     print(min_max_acc)
            # c,p= pearsonr(model_acc, answer)
            
            
            # correlations[curriculum].append(c)
        #     if p <0.05:
        #         correlations[curriculum].append(c)
        #     else:
        #         correlations[curriculum].append(0)
    # print(correlations)
    for c in np.unique(data_sns['curriculum']):
        idx = np.where(np.array(data_sns['curriculum']) == c)[0]
        r,p= pearsonr(
            np.array(data_sns['perceived'])[idx],
            np.array(data_sns['actual'])[idx])
        res = stats.spearmanr(
            np.array(data_sns['perceived'])[idx],
            np.array(data_sns['actual'])[idx])
        print(c, r , p)

        sns.regplot( x = np.array(data_sns['perceived'])[idx], y = np.array(data_sns['actual'])[idx])
        plt.title(c+':'+' corr={}'.format(str(round(r,3)))+' p={}'.format(str(round(p,5))))
        plt.xlabel('perceived accuracy')
        plt.ylabel('actual accuracy')
        
        plt.savefig('correlation_questionnaire_{}.pdf'.format(c))
        plt.show()
    with open('correlation.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
            keys = data_sns.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(data_sns['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = data_sns[k][l]
                w.writerow(row_to_write)

    # mean_corr = []
    # for cond in correlations.keys():
    #     mean_corr.append(np.mean(correlations[cond]))
    # return mean_corr

def learning_rate(data,models):
    
    data_sns = {'pid': [], 'condition': [], 'lr': []}
    for pid in data.keys():
        
        acc = []
        X = np.array(data[pid]['training']['x'])
        y = np.array(data[pid]['training']['y'])
        for trial in range(17,len(X)):        
            classifier = models[pid]['training'][trial]
            acc.append(1-classifier.score(
                            np.array(data[pid]['posttest']['x']),
                            np.array(data[pid]['posttest']['y'])))
        # print(len(np.arange(17,len(X))))
        # print(len(acc))
        data_sns['pid'].append(pid)
        data_sns['condition'].append(data[pid]['cond'])
        slope, intercept, r, p, std_err = stats.linregress(np.log(np.arange(17,len(X))), np.log(acc))
        data_sns['lr'].append(slope)
        with open('lr.csv', 'w') as f: 
            keys = data_sns.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(data_sns['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = data_sns[k][l]
                w.writerow(row_to_write)

def compute_pos_neg_recall(data, 
                           models, 
                           train_phase, 
                           model_t, 
                           perclass,
                           type_of_plot):
    
    # depending if we consider per-class accuracy or mean accuracy over classes
    if not perclass: 
        data_sns = {'pid': [], 'condition': [], 'recall': []}
    else: 
        data_sns = {'pid': [], 'condition': [], 'class': [], 'recall': []}

    for pid in data.keys():
        
        if models == None:
            classifier = LinearDiscriminantAnalysis()
            X = np.array(data[pid][train_phase]['x'])[:model_t,:]
            y = np.array(data[pid][train_phase]['y'])[:model_t]
            classifier.fit(X, y)
        else:
            classifier = models[pid]['training'][model_t]
        
        condition = data[pid]['cond']
        actual = []
        preds = []
        for t in ['positive', 'negative']:
            preds.extend(classifier.predict(np.array(data[pid][t]['x'])))
            actual.extend(np.array(data[pid][t]['y']))
        # print(len(preds),len(actual))
        if not perclass:  
            tp = 0
            fn = 0
            data_sns['pid'].append(pid)
            data_sns['condition'].append(condition)
            for i in range (len(actual)):
                if(actual[i] != preds[i]):
                    fn += 1
                else: tp += 1
            # cm = confusion_matrix(actual, preds, labels=classifier.classes_).T
            # print(cm)
            print(tp,fn)
            data_sns['recall'].append(tp/(fn+tp))
        else:
            perclass_rec = perclass_recall(
                actual,
                preds, 
                classifier.classes_
            )
            for k in perclass_rec.keys():
                data_sns['pid'].append(pid)
                data_sns['condition'].append(data[pid]['cond'])
                data_sns['class'].append(k)
                data_sns['recall'].append(perclass_rec[k])

    with open('recall_perclass={}.csv'.format(perclass), 'w') as f:  # You will need 'wb' mode in Python 2.x
        keys = data_sns.keys()
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for l in range(len(data_sns['pid'])):
            row_to_write = {}
            for k in keys:
                # if k == 'condition':
                #     row_to_write[k] = conds_tab.index(data_sns[k][l])
                # elif k == 'type':
                #     row_to_write[k] = type_tab.index(data_sns[k][l])
                # else:
                #     row_to_write[k] = data_sns[k][l]
                row_to_write[k] = data_sns[k][l]
            # if row_to_write['type'] == 0:
            w.writerow(row_to_write)


    df = pd.DataFrame(data_sns)
    print('** one-way ANOVA')
    model = ols('recall ~ C(condition)', data=df).fit()
    print(sm.stats.anova_lm(model, typ=1))    
    print('** pairwise test')
    for ci, c in enumerate(np.unique(data_sns['condition'])):
        for c2i, c2 in enumerate(np.unique(data_sns['condition'])):
            if c2i > ci:
                print('{} - {}\t'.format(c,c2), 
                        stats.ttest_ind(
                        df[(df['condition'] == c)]['recall'], 
                        df[(df['condition'] == c2)]['recall']))
    if type_of_plot== 'barplot':
        sns.barplot(data=df, x='condition', y='recall')
    elif type_of_plot== 'violinplot':
        sns.violinplot(data=df, x='condition', y='recall', inner="points")
    # plt.title("retrained models - anova's pvalue={:.3f}".format(pval))
    plt.savefig('recall-analysis_plot={}_perclass={}.png'.format(type_of_plot, perclass))
    plt.show()

if __name__ == "__main__":
    # folder path
    dir_path = 'data'
    data, models = load_data(dir_path)
    # questionnaires('./Questionnaire post entrainement.csv',data,models)
    # learning_rate(data,models)
    recall_pos_neg = {
        "train_phase": 'training', 
        "model_t": 120,
        "perclass": True,
        "type_of_plot": 'barplot'
    }
    compute_pos_neg_recall(data, 
        models=models,
        **recall_pos_neg)