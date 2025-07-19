import csv
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns 
import pandas as pd 
from scipy import stats

maps = {'NUC': 'RC', 'UC': 'UCC', 'NUCS': 'GSC'}


def model_accuracy_plots():
    
    plt.figure(figsize=(6,6))
    # sns.color_palette("flare", as_cmap=True)
    sns.color_palette("rocket")
    accs_t = pd.read_csv('accuracy_along_training.csv')
    # accs_pt = pd.read_csv('accuracies_posttest.csv')
    # accs_tr = pd.read_csv('accuracies_training.csv')

    # plt.figure(figsize=(12,5))
    # print(accs_t[accs_t['trial'] == 103])
    ax2 = plt.subplot(1,1,1)
    sns.barplot(
        data=accs_t[accs_t['trial'] == 103], 
        x='condition', 
        y='accuracy', 
        palette='Set2', 
        capsize=.1, 
        edgecolor=".5",)
    ax2.set_xticklabels([maps[l.get_text()] for l in ax2.get_xticklabels()], fontsize=13)
    plt.ylabel('$\Delta$(Accuracy)', fontsize=13)
    plt.xlabel('')

    # ax1 = plt.subplot(1,2,2)
    # sns.barplot(data=accs_pt, x='condition', y='accuracy', palette='Set2', capsize=.1, edgecolor=".5",)
    # ax1.set_xticklabels([maps[l.get_text()] for l in ax1.get_xticklabels()], fontsize=13)
    # plt.title('Test Set = Posttest Data')
    # plt.ylabel('Accuracy', fontsize=13)
    # plt.xlabel('')
    
    plt.savefig('model_accuracies.pdf')
    plt.show()

    ax = plt.subplot(1,1,1)
    sns.lineplot(
        data=accs_t, 
        x='trial', 
        y='accuracy', 
        hue='condition', 
        palette='Set2', 
        hue_order=['NUC', 'NUCS', 'UC'])
    ax.legend(labels=['RC', 'GSC', 'UCC'], title="curriculum", fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.xlabel('Trial', fontsize=13)
    plt.savefig('accuracy_along_training_condition.pdf')
    plt.show()



def model_delta_accuracy_plots():
    
    plt.figure(figsize=(6,6))
    # sns.color_palette("flare", as_cmap=True)
    sns.color_palette("rocket")
    accs_t = pd.read_csv('accuracy_along_training_offset.csv')
    
    # data2 = {'pid': [], 'condition': [], 'class': [], 'deltaAcc': []}
    # for i in range(len(accs_t[accs_t['trial'] == 103]['accuracy'])):
    #     data2['pid'].append(accs_t[accs_t['trial'] == 103]['pid'].iloc[i])
    #     data2['condition'].append(accs_t[accs_t['trial'] == 103]['condition'].iloc[i])
    #     data2['class'].append(accs_t[accs_t['trial'] == 103]['class'].iloc[i])
    #     data2['deltaAcc'].append(
    #         accs_t[accs_t['trial'] == 103]['accuracy'].iloc[i] - accs_t[accs_t['trial'] == 1]['accuracy'].iloc[i])
    # df2 = pd.DataFrame(data2)

    # with open('delataaccuracy.csv', 'w') as f: 
    #     keys = data2.keys()
    #     w = csv.DictWriter(f, keys)
    #     w.writeheader()
    #     for l in range(len(data2['pid'])):
    #         row_to_write = {}
    #         for k in keys:
    #             row_to_write[k] = data2[k][l]
    #         w.writerow(row_to_write)

    ax2 = plt.subplot(1,1,1)
    sns.barplot(
        data=accs_t[accs_t['trial'] == 103], 
        x='condition', 
        y='accuracy', 
        palette='Set2', 
        capsize=.1, 
        edgecolor=".5",)
    ax2.set_xticklabels([maps[l.get_text()] for l in ax2.get_xticklabels()], fontsize=13)
    plt.ylabel('$\Delta$(Accuracy)', fontsize=13)
    plt.xlabel('Condition', fontsize=13)
    # plt.xlabel('')

    # ax1 = plt.subplot(1,2,2)
    # sns.barplot(data=accs_pt, x='condition', y='accuracy', palette='Set2', capsize=.1, edgecolor=".5",)
    # ax1.set_xticklabels([maps[l.get_text()] for l in ax1.get_xticklabels()], fontsize=13)
    # plt.title('Test Set = Posttest Data')
    # plt.ylabel('Accuracy', fontsize=13)
    # plt.xlabel('')
    
    plt.savefig('model_deltaaccuracies.pdf')
    plt.show()


def model_posneg():
    
    plt.figure(figsize=(6,6))

    # sns.color_palette("flare", as_cmap=True)
    sns.color_palette("rocket")
    accs_t = pd.read_csv('pos_neg.csv')
    # accs_pt = pd.read_csv('accuracies_posttest.csv')
    # accs_tr = pd.read_csv('accuracies_training.csv')

    # plt.figure(figsize=(12,5))
    # print(accs_t[accs_t['trial'] == 103])
    ax = plt.subplot(1,1,1)
    sns.barplot(
        data=accs_t, 
        x='condition', 
        y='rate', 
        hue='type',
        palette='Set2', 
        capsize=.1, 
        edgecolor=".5")
    ax.set_xticklabels([maps[l.get_text()] for l in ax.get_xticklabels()], fontsize=13)
    plt.ylabel('Rates', fontsize=13)
    plt.xlabel('Condition', fontsize=13)
    # plt.legend(fontsize=13)
    ax.legend(title="", fontsize=14)
    # ax1.set_xlabel('X-axis') 
    # ax1.set_ylabel('True Positive Rate', color = 'red') 
    # # ax1.plot(x, data_1, color = 'red') 
    # ax1.tick_params(axis ='y', labelcolor = 'red') 

    # ax2 = ax1.twinx() 
    # ax2.set_ylabel('True Negative Rate', color = 'blue') 
    # # ax2.plot(x, data_2, color = 'blue') 
    # ax2.tick_params(axis ='y', labelcolor = 'blue') 
    plt.savefig('pos_neg.pdf')
    plt.show()


def lr_plot():
    
    plt.figure(figsize=(6,6))

    data = pd.read_csv('lr.csv') 
    data_df = pd.DataFrame(data)
    # data_df.to_csv('separability_start_end_posttest', index=False)
    # data_df.set_index('phase').loc['training-start', 'training-end', 'posttest']

    for ci, c in enumerate(['NUC', 'UC', 'NUCS']):
        for c2i, c2 in enumerate(['NUC', 'UC', 'NUCS']):
            if c2i > ci:
                print(c,c2, stats.ttest_ind(data_df[data_df['condition'] == c]['lr'], data_df[data_df['condition'] == c2]['lr']))

    ax = plt.subplot(1,1,1) #plt.subplots(figsize=(6, 5), tight_layout=True)
    sns.barplot(
        data=data_df, 
        x='condition', 
        y='lr', palette='Set2', capsize=.1, edgecolor=".5", order=['NUC', 'NUCS', 'UC'])
    # ax.set_xticklabels([maps[l.get_text()] for l in ax.get_xticklabels()], fontsize=13)
    # ax = df.set_index(field).loc[day_order].plot(kind="bar", legend=False)
    # ax.legend(labels=['RC', 'GSC', 'UCC'], title="curriculum", fontsize=13)
    ax.set_xticklabels([maps[l.get_text()] for l in ax.get_xticklabels()], fontsize=13)
    plt.ylabel('Learning Rate', fontsize=13)
    plt.xlabel('Condition', fontsize=13)
    # plt.ylim([0.0, 3.0])
    # plt.title('Gesture class separability', fontsize=15)
    
    plt.savefig('learning_rate_plot.pdf')
    plt.show()




def separability_plot():
    
    sep_training = pd.read_csv('consistency_along_training.csv')
    sep_posttest = pd.read_csv('consistency_posttest.csv') 

    sep_training_first = sep_training[sep_training['trial'] == 1]
    sep_training_last = sep_training[sep_training['trial'] == 103]

    print(sep_training_first)

    keys = ['pid', 'condition', 'class', 'consistency', 'phase']
    data = {}
    for k in keys:
        data[k] = []
    for r in range(len(sep_posttest['pid'])):
        for k in ['pid', 'condition', 'class', 'consistency']:
            if k == 'condition':
                data[k].append(maps[sep_posttest[k].iloc[r]])
            else:
                data[k].append(sep_posttest[k].iloc[r])
        data['phase'].append('posttest')
    for r in range(len(sep_training_first['pid'])):
        for k in ['pid', 'condition', 'class', 'consistency']:
            if k == 'condition':
                data[k].append(maps[sep_training_first[k].iloc[r]])
            else:
                data[k].append(sep_training_first[k].iloc[r])
        data['phase'].append('training-start')
    for r in range(len(sep_training_last['pid'])):
        for k in ['pid', 'condition', 'class', 'consistency']:
            if k == 'condition':
                data[k].append(maps[sep_training_last[k].iloc[r]])
            else:
                data[k].append(sep_training_last[k].iloc[r])
        data['phase'].append('training-end')
    
    data_df = pd.DataFrame(data)
    data_df.to_csv('separability_start_end_posttest', index=False)
    # data_df.set_index('phase').loc['training-start', 'training-end', 'posttest']
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    sns.barplot(
        data=data_df, 
        x='phase', 
        y='consistency', hue='condition', palette='Set2', capsize=.1, edgecolor=".5", order=['training-start', 'training-end', 'posttest'])
    # ax.set_xticklabels([maps[l.get_text()] for l in ax.get_xticklabels()], fontsize=13)
    # ax = df.set_index(field).loc[day_order].plot(kind="bar", legend=False)
    # ax.legend(labels=['RC', 'GSC', 'UCC'], title="curriculum", fontsize=13)
    plt.ylabel('Separability', fontsize=13)
    plt.xlabel('Phase', fontsize=13)
    plt.ylim([0.0, 3.0])
    # plt.title('Gesture class separability', fontsize=15)
    
    plt.savefig('separability_plots.pdf')
    plt.show()

    # # sns.color_palette("flare", as_cmap=True)
    # sep_training = pd.read_csv('separability_along_training.csv')
    
    # fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    # sns.lineplot(data=sep_training, x='trial', y='separability', hue='condition', palette='Set2')
    # # ax.set_xticklabels([maps[l.get_text()] for l in ax.get_xticklabels()], fontsize=13)
    # ax.legend(labels=['RC', 'GSC', 'UCC'], title="curriculum", fontsize=13)
    # plt.ylabel('Separability', fontsize=13)
    # plt.xlabel('Trial num.', fontsize=13)
    # plt.ylim([0.0, 2.5])
    # plt.title('Gesture class separability during training', fontsize=15)

    # plt.savefig('separability_plot_training.pdf')
    # plt.show()
    

    # fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    # sep_training_last = sep_training[sep_training['trial'] == 103]
    # sns.barplot(data=sep_training_last, x='condition', y='separability', palette='Set2', capsize=.1, edgecolor=".5",)
    # ax.set_xticklabels([maps[l.get_text()] for l in ax.get_xticklabels()], fontsize=13)
    # plt.ylabel('Separability', fontsize=13)
    # plt.xlabel('Curriculum', fontsize=13)
    # plt.ylim([0.0, 2.5])
    # plt.title('Gesture class separability at trial 104', fontsize=15)
    
    # plt.savefig('separability_plot_training_last.pdf')
    # plt.show()


    # sep_posttest = pd.read_csv('separability_posttest.csv')

    # fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    # sns.barplot(data=sep_posttest, x='condition', y='separability', palette='Set2', capsize=.1, edgecolor=".5",)
    # ax.set_xticklabels([maps[l.get_text()] for l in ax.get_xticklabels()], fontsize=13)
    # plt.ylabel('Separability', fontsize=13)
    # plt.xlabel('Curriculum', fontsize=13)
    # plt.ylim([0.0, 2.5])
    # plt.title('Gesture class separability at posttest', fontsize=15)

    # # # ax1 = plt.subplot(1,2,2)
    # # # sns.barplot(data=accs_pt, x='condition', y='accuracy', palette='Set2', capsize=.1, edgecolor=".5",)
    # # # ax1.set_xticklabels([maps[l.get_text()] for l in ax1.get_xticklabels()], fontsize=13)
    # # # plt.title('Test Set = Posttest Data')
    # # # plt.ylabel('Accuracy', fontsize=13)
    # # # plt.xlabel('')
    
    # plt.savefig('separability_plot_posttest.pdf')
    # plt.show()

# model_posneg()
# model_accuracy_plots()
# model_delta_accuracy_plots()
separability_plot()
# lr_plot()