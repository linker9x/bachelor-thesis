import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re


def plot_file(csv_in, dir_out):

    with open(csv_in, 'r') as f:
        lines = f.readlines()

    head_toks = lines[0].strip().split(',')
    headers = head_toks[1:]
    headers.remove(head_toks[len(head_toks) - 2])

    train_dict = dict()
    test_dict = dict()
    for i in range(1, len(lines)):
        line = lines[i].strip()
        toks = line.split(',')
        dataset = toks[0]
        process_dict = train_dict
        if dataset.find('_test_') >= 0:
            process_dict = test_dict
        # fold_idx = int(dataset[len(dataset) - 5])
        dataset = re.sub('\_test_([0-4]+)\.csv', '', dataset)
        dataset = re.sub('\_train_([0-4]+)\.csv', '', dataset)
        if dataset not in process_dict:
            process_dict[dataset] = list()

        for j in range(1, len(toks)):
            if j == len(toks) - 2:
                continue
            if (j < len(toks) -2 and len(process_dict[dataset]) < j) or (j > len(toks) - 2 and len(process_dict[dataset]) < j-1): # remove ClsCoef from the plots
                process_dict[dataset].append(list())
            # if len(process_dict[dataset]) < j:
            #     process_dict[dataset].append(list())
            process_dict[dataset][j-1 if j < len(toks) - 2 else j-2].append(float(toks[j]) if j != len(toks) - 3 else float(toks[j])/1000)# if j < len(toks) - 3 else j-2

    for dataset in train_dict.keys():
        plot_out_path = dir_out + dataset + '.png'
        columns = list()
        for header in headers:
            columns.append(header + '_train')
            columns.append(header + '_test')
        data = list()
        for i, vals in enumerate(train_dict[dataset]):
            data.append(vals)
            data.append(test_dict[dataset][i])

        df = pd.DataFrame(np.asmatrix(data), index=columns)
        df = df.T

        fig = plt.figure()
        ax = fig.add_subplot(111)
        df.boxplot(figsize=(19,10),
                        whis=[5,95], return_type='axes')#grid=False, rot=45,
        # plt.ylabel('Complexity score', fontsize =18)
        # plt.xlabel(dataset, fontsize =18)

        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()

        plt.savefig(plot_out_path,bbox_inches='tight',dpi=600) # can be 1200
        plt.close()


dir_in = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/fold_data/'
csv_in = dir_in + 'test.csv'
csv_in = dir_in + 'fold_complexity_modified.csv'
#csv_in = dir_in + 'chin_fold_complexity.csv'
dir_out = dir_in + 'complexity_plots/'
plot_file(csv_in, dir_out)