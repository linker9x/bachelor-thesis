import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def read_sum_file(csv_path, out_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    writer = open(out_path, 'w')

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
        itok = 3
        while itok < len(tokens):
            iname = (itok - 3) / 2
            fs_clf = fs_alg_names[iname] + '#' + clf_name
            if clf_name != 'reconstruction':
                if fs_clf not in sum_dict:
                    sum_dict[fs_clf] = dict()
                if num_feat != 0:
                    sum_dict[fs_clf][num_feat] = round(float(tokens[itok]),2)
                else:
                    baseline_dict[fs_clf] = round(float(tokens[itok]),2)
            else:
                if fs_alg_names[iname] not in reconstruction_dict:
                    reconstruction_dict[fs_alg_names[iname]] = dict()
                if num_feat != 0:
                 reconstruction_dict[fs_alg_names[iname]][num_feat] = round(float(tokens[itok]),2)
            itok += 2
    writer.write('fs_clf_name,baseline,max_f1,min_f1,median_f1,mean_f1,std_f1,max_percent,percent_above_baseline,percent_nearly_baseline\n')

    largest_mean = -1
    largest_mean_entry = ''
    smallest_std = 100
    smallest_std_entry = ''
    earlest_above = -1
    earlest_above_entry = ''
    earlest_nearly = -1
    earlest_nearly_entry = ''
    max_f1 = -1
    max_entry = ''
    res_dict = dict()
    columns = ['baseline','max','min','median','mean','std']
    idx_cols = ['max_idx','i_above','i_nearly']
    neural_data = list()
    svm_data = list()
    neural_idx = list()
    svm_idx = list()
    indexes = []
    for fs_clf_name in sorted(sum_dict.keys()):
        val_list = list()
        writer.write(fs_clf_name + ',' + str(baseline_dict[fs_clf_name]) + ',')

        for ifeat in sorted(sum_dict[fs_clf_name].keys()):
            val_list.append(sum_dict[fs_clf_name][ifeat])

        arr = np.array(val_list)
        max = np.amax(arr)
        writer.write("{:.2f}".format(max) + ',')
        min = np.amin(arr)
        writer.write("{:.2f}".format(min) + ',')
        median = np.median(arr)
        writer.write("{:.2f}".format(median) + ',')
        mean = np.mean(arr)
        writer.write("{:.2f}".format(mean) + ',')
        std = np.std(arr)
        writer.write("{:.2f}".format(std) + ',')
        # print(np.where(arr == np.amax(arr))[0][0])
        max_idx = np.where(arr == max)[0][0] + 1
        writer.write(str(max_idx) + ',')
        i_above = -1
        i_nearly = -1
        for i,score in enumerate(val_list):
            if i_above == -1 and score >= baseline_dict[fs_clf_name]:
                i_above = i + 1
            if i_nearly == -1 and score <= baseline_dict[fs_clf_name] and abs(score - baseline_dict[fs_clf_name]) < 1:
                i_nearly = i + 1
        writer.write(str(i_above) + ',' + str(i_nearly) + '\n')

        if max_f1 < max:
            max_entry = fs_clf_name + ':' + str(max_idx)
            max_f1 = max
        elif max_f1 == max:
            max_entry += ', ' + fs_clf_name + ':' + str(max_idx)
        if largest_mean < mean:
            largest_mean = mean
            largest_mean_entry = fs_clf_name
        if smallest_std > std:
            smallest_std = std
            smallest_std_entry = fs_clf_name
        if i_above != -1 and (earlest_above == -1 or earlest_above >= i_above):
            if earlest_above > i_above:
                earlest_above_entry = fs_clf_name
                earlest_above = i_above
            elif earlest_above == i_above:
                earlest_above_entry += ', ' + fs_clf_name
            if earlest_nearly == -1:
                earlest_above = i_above
        if i_nearly != -1 and (earlest_nearly == -1 or earlest_nearly >= i_nearly):
            if earlest_nearly > i_nearly:
                earlest_nearly_entry = fs_clf_name
                earlest_nearly = i_nearly
            elif earlest_nearly == i_nearly:
                earlest_nearly_entry += ', ' + fs_clf_name
            if earlest_nearly == -1:
                earlest_nearly = i_nearly

        stat_list= [baseline_dict[fs_clf_name],max,min,median,mean,std]
        i_stats = [max_idx,i_above if i_above != -1 else 0,i_nearly if i_nearly != -1 else 0]
        if fs_clf_name.endswith('#Neural Net'):
            neural_data.append(stat_list)
            neural_idx.append(i_stats)
            indexes.append(fs_clf_name.replace('#Neural Net','').replace('TSFS_my_autoencoder', 'TSFS').replace('TSFS_MCFS', 'MCFS'))
        else:
            svm_data.append(stat_list)
            svm_idx.append(i_stats)
        # data.append(stat_list)

    svm_df = pd.DataFrame(np.asmatrix(svm_data), index=indexes, columns=columns)
    plot_out_path = csv_path.replace('.csv','_SVM.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    svm_df.plot(kind='bar')#,stacked=True
    plt.legend()
    plt.savefig(plot_out_path,bbox_inches='tight',dpi=100)
    plt.close()

    svm_df = pd.DataFrame(np.asmatrix(neural_data), index=indexes, columns=columns)
    plot_out_path = csv_path.replace('.csv','_Neural.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    svm_df.plot(kind='bar')#,stacked=True
    plt.legend(fontsize='small',loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(plot_out_path,bbox_inches='tight',dpi=100)
    plt.close()

    svm_df = pd.DataFrame(np.asmatrix(svm_idx), index=indexes, columns=idx_cols)
    plot_out_path = csv_path.replace('.csv','_SVM_idx.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    svm_df.plot(kind='bar')#,stacked=True
    plt.legend()
    plt.savefig(plot_out_path,bbox_inches='tight',dpi=100)
    plt.close()

    svm_df = pd.DataFrame(np.asmatrix(neural_idx), index=indexes, columns=idx_cols)
    plot_out_path = csv_path.replace('.csv','_Neural_idx.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    svm_df.plot(kind='bar')#,stacked=True
    plt.legend()
    plt.savefig(plot_out_path,bbox_inches='tight',dpi=100)
    plt.close()

    writer.write('\n\nIN SUM:\n')
    writer.write('max_f1 value:, ' + "{:.2f}".format(max_f1) + ', acquire by:, ' + max_entry + '\n')
    writer.write('largest mean value:, ' + "{:.2f}".format(largest_mean) + ' ,acquire by:, ' + largest_mean_entry + '\n')
    writer.write('min std value:, ' + "{:.2f}".format(smallest_std) + ', acquire by: ,' + smallest_std_entry + '\n')
    writer.write('earlest above value: ,' + str(earlest_above) + ' ,acquire by: ,' + earlest_above_entry + '\n')
    writer.write('earlest nearly value:, ' + str(earlest_nearly) + ', acquire by:, ' + earlest_nearly_entry + '\n')


    writer.write('\n\nReconstruction error:\n')
    writer.write(',max,min,median,mean,std,min_idx\n')

    res_col = ['max','min','median','mean','std']
    indexes = []
    res_data = []
    for fs_clf_name in reconstruction_dict.keys():
        val_list = list()
        for ifeat in sorted(reconstruction_dict[fs_clf_name].keys()):
            val_list.append(reconstruction_dict[fs_clf_name][ifeat])

        writer.write(fs_clf_name + ',')
        arr = np.array(val_list)
        max = np.amax(arr)
        min = np.amin(arr)
        median = np.median(arr)
        writer.write("{:.2f}".format(max) + ',')
        writer.write("{:.2f}".format(min) + ',')
        writer.write("{:.2f}".format(median) + ',')
        mean = np.mean(arr)
        writer.write("{:.2f}".format(mean) + ',')
        std = np.std(arr)
        writer.write("{:.2f}".format(std) + ',')
        # print(np.where(arr == np.amax(arr))[0][0])
        min_idx = np.where(arr == np.amin(arr))[0][0] + 1

        writer.write(str(min_idx) + '\n')
        stat_list = [max, min, median, mean, std]
        indexes.append(fs_clf_name.replace('TSFS_my_autoencoder', 'TSFS').replace('TSFS_MCFS', 'MCFS'))
        res_data.append(stat_list)


    svm_resconstruction_df = pd.DataFrame(np.asmatrix(res_data), index=indexes, columns=res_col)
    plot_out_path = csv_path.replace('.csv','_reconstruction.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    svm_resconstruction_df.plot(kind='bar')#,stacked=True
    plt.legend()
    plt.savefig(plot_out_path,bbox_inches='tight',dpi=100)
    plt.close()


    writer.close()
    return {dataset: res_dict}

def draw_plots(dataset, plot_dir):
    return


def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir


def join_all(dir):
    subdirs = os.listdir(dir)
    dir = standardize_dir(dir)
    for subdir in subdirs:
        if not os.path.isdir(dir + subdir):
            continue
        # join_dir(dir + subdir + '/', subdir)



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

read_sum_file(result_dir + 'Prostate_GE_sum_result.csv', result_dir + 'Prostate_GE_stats.csv')