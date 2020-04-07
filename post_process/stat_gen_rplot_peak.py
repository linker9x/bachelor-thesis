import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



def read_sum_file(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    header_toks = lines[0].strip().split(',')
    itok = 3
    fs_alg_names = list()
    while itok < len(header_toks):
        fs_alg_names.append(header_toks[itok][:len(header_toks[itok])-7])
        itok += 2
    sum_dict = dict()
    reconstruction_dict = dict()
    dataset = ''
    baseline_dict = dict()
    # skip the header line
    for i in range(1, len(lines)):
        line = lines[i]
        line = line.strip()
        if line == '':
            continue
        tokens = line.split(',')
        dataset = tokens[0]
        num_feat = int(tokens[1])

        clf_name = tokens[2]
        if clf_name == 'LinearSVM':
            continue
        itok = 3
        while itok < len(tokens):
            iname = (itok - 3) / 2
            fs_name = fs_alg_names[iname]
            if clf_name not in sum_dict:
                sum_dict[clf_name] = dict()

            if fs_name not in sum_dict[clf_name]:
                sum_dict[clf_name][fs_name] = dict()
            if num_feat != 0:
                sum_dict[clf_name][fs_name][num_feat] = round(float(tokens[itok]),2)
            elif clf_name not in baseline_dict:
                baseline_dict[clf_name] = round(float(tokens[itok]),2)
            if clf_name == 'reconstruction':
                if fs_name not in reconstruction_dict:
                    reconstruction_dict[fs_name] = dict()
                if num_feat != 0:
                 reconstruction_dict[fs_name][num_feat] = round(float(tokens[itok]),2)
            itok += 2

    index_dict = dict()

    for clf_name in sum_dict.keys():
        fs_dict = sum_dict[clf_name]
        if clf_name not in index_dict:
            index_dict[clf_name] = dict()
        for fs_name in sorted(fs_dict.keys()):
            val_list = list()

            for ifeat in sorted(fs_dict[fs_name].keys()):
                val_list.append(fs_dict[fs_name][ifeat])

            arr = np.array(val_list)
            max_idx = np.where(arr == np.amax(arr))[0][0] + 1
            if clf_name == 'reconstruction':
                max_idx = np.where(arr == np.min(arr))[0][0] + 1
            i_above = 0
            i_nearly = 0
            for i,score in enumerate(val_list):
                if i_above == 0 and score >= baseline_dict[clf_name]:
                    i_above = i + 1
                if i_nearly == 0 and score <= baseline_dict[clf_name] and abs(score - baseline_dict[clf_name]) < 1:
                    i_nearly = i + 1
            index_dict[clf_name][fs_name]= (max_idx, i_above, i_nearly)

    return index_dict

def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir


def gen_rplot_all(dir):
    files = os.listdir(dir)
    dir = standardize_dir(dir)
    sum_dict = dict()
    for file in files:
        if not file.endswith('_sum_result.csv'):
            continue

        dataset = file.replace('_sum_result.csv', '')
        if dataset == 'Data_Cortex_Nuclear':
            continue
        sum_dict[dataset] = read_sum_file(dir + file)
    # print(sum_dict)

    max_dict = dict()
    above_dict = dict()
    nearly_dict = dict()
    fs_list = []
    dataset_list = sum_dict.keys()
    dataset_dict= dict()
    for i_data, dataset in enumerate(sum_dict.keys()):
        for clf_name in sum_dict[dataset]:
            if clf_name not in dataset_dict:
                dataset_dict[clf_name] = list()
            dataset_dict[clf_name].append(list())
            if clf_name not in max_dict:
                max_dict[clf_name] = list()
                nearly_dict[clf_name] = list()
                above_dict[clf_name] = list()

            fs_list = sorted(sum_dict[dataset][clf_name])
            for i, fs in enumerate(fs_list):
                if len(max_dict[clf_name]) <= i:
                    max_dict[clf_name].append(list())
                    above_dict[clf_name].append(list())
                    nearly_dict[clf_name].append(list())
                dataset_dict[clf_name][i_data].append(sum_dict[dataset][clf_name][fs][0])
                max_dict[clf_name][i].append(sum_dict[dataset][clf_name][fs][0])
                above_dict[clf_name][i].append(sum_dict[dataset][clf_name][fs][1])
                nearly_dict[clf_name][i].append(sum_dict[dataset][clf_name][fs][2])


    out_dir = dir + 'index_plots/'
    for clf_name in dataset_dict.keys():
        plot_out_path = out_dir + clf_name + '_max.png'
        plot_data = np.asarray(max_dict[clf_name])
        print(plot_data)
        plot_df = pd.DataFrame(plot_data, index=fs_list)
        plot_df = plot_df.T
        fig = plt.figure()

        ax = fig.add_subplot(111)
        plt.ylim(0, 20)
        plot_df.boxplot(return_type='axes')#grid=False, rot=45,
        plt.ylabel('F1 score')
        plt.xlabel('Feature selection methods')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.savefig(plot_out_path,bbox_inches='tight',dpi=600)
        plt.close()

        # plot_out_path = out_dir + clf_name + '_dataset.png'
        # plot_data = np.asarray(dataset_dict[clf_name])
        # plot_df = pd.DataFrame(plot_data, index=dataset_list)
        # plot_df = plot_df.T
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plot_df.boxplot(figsize=(19,10),
        #                 whis=[5,95], return_type='axes')#grid=False, rot=45,
        # plt.ylabel('F1 score')
        # plt.xlabel('Feature selection methods')
        # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        # plt.savefig(plot_out_path,bbox_inches='tight',dpi=600)
        # plt.close()

dir_in = '/Users/ngandong/Desktop/feature_selection/results/normalize/chin/'
# join_dir(dir_in, 'chin')

dir_in = '/Users/ngandong/Desktop/feature_selection/results/normalize_tsfs/'
dir_in = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/normalize_tsfs/'
# join_all(dir_in)
# hadoop_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/hadoop_res/'
# dl_in = hadoop_dir + 'dl/normalize/'
# dl_percent_in = hadoop_dir + 'percentage/normalize/'
# non_dl_in = hadoop_dir + 'non_dl/normalize/'
# non_dl_percent_in = hadoop_dir + 'non_dl_percentage/normalize/'
# join_all(dl_in)
# join_all(dl_percent_in)
# join_all(non_dl_in)
# join_all(non_dl_percent_in)

# test_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/result/CNS/'
# join_dir(test_dir, 'CNS')
result_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/result_Jul9/'
result_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/result_Jul9_update_starting_points/'

# result_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/normalize_tsfs/'
# result_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/hadoop_res/percentage/normalize/'
# join_all(result_dir)

# read_sum_file(result_dir + 'Prostate_GE_sum_result.csv', result_dir + 'Prostate_GE_stats.csv')
gen_rplot_all(result_dir)