import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def join_file(csv_path, fs_alg):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    # skip the header line
    res_dict = dict()
    for i in range(1, len(lines)):
        line = lines[i]
        line = line.strip()
        if line == '':
            continue
        tokens = line.split(',')
        start_idx = 1
        if not tokens[0].isdigit():
            start_idx = 2
        num_feat = str(int(float(tokens[start_idx]))) # if tokens[start_idx] != '0.0' else '0'
        clf_name = tokens[start_idx + 1].replace('Neural Net', 'NeuralNet').replace('Linear SVM', 'LinearSVM')
        acc = float(tokens[start_idx + 2])
        pre = float(tokens[start_idx + 3])
        rec = float(tokens[start_idx + 4])
        f1 = float(tokens[start_idx + 5].strip())

        if num_feat not in res_dict:
            res_dict[num_feat] = dict()
        num_feat_dict = res_dict[num_feat]
        if clf_name not in num_feat_dict:
            num_feat_dict[clf_name] = list()
        if clf_name != 'reconstruction':
            num_feat_dict[clf_name].append([acc, pre, rec, f1])
        else:
            num_feat_dict[clf_name].append([f1, pre, rec, acc])

    alg_per_dict = dict()
    for i, num_feat in enumerate(sorted(res_dict.keys())):
        for clf in sorted(res_dict[num_feat].keys()):
            n_steps = 20 #len(res_dict.keys())
            if clf not in alg_per_dict:
                alg_per_dict[clf] = np.zeros((n_steps, 3))
            matrix = np.asmatrix(res_dict[num_feat][clf])
            mean = matrix.mean(0).tolist()
            std = matrix.std(0).tolist()
            res_dict[num_feat][clf] = [mean[0],std[0]]
            idx = int(num_feat[:len(num_feat)-2] if '.' in num_feat else num_feat)
            if idx > n_steps - 1:
                continue
            alg_per_dict[clf][idx,0] = num_feat
            alg_per_dict[clf][idx,1] = mean[0][3]
            alg_per_dict[clf][idx,2] = std[0][3]

    return {fs_alg: res_dict}, alg_per_dict


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
        join_dir(dir + subdir + '/', subdir)

def join_dir(dir_in, dataset):
    files = os.listdir(dir_in)
    dir_in = standardize_dir(dir_in)

    out_path = dir_in[:len(dir_in) - 1] + '_sum_result.csv'
    writer = open(out_path, 'w')

    idx = 0
    all_res = dict()
    plot_data_dict = dict()
    for file in files:
        if not file.endswith('.csv') or file == 'sum_result.csv':
            continue
        fs_alg = file.replace('.csv', '').replace('TSFS_my_autoencoder', 'TSFS').replace('TSFS_MCFS', 'MCFS').replace('elastic_net', 'enet')
        new_entry, alg_per_dict = join_file(dir_in + file, fs_alg)
        all_res.update(new_entry)
        idx += 1

        for clf_name in alg_per_dict.keys():
            if clf_name not in plot_data_dict:
                plot_data_dict[clf_name] = dict()
            plot_data_dict[clf_name][fs_alg] = alg_per_dict[clf_name]

    # write result to file
    result_str = dict()
    no_fs = len(all_res.keys())
    for i, fs_alg in enumerate(all_res.keys()):
        # insert header line
        if i == 0:
            writer.write('dataset,num_feat,clf_name')
            for ifs in range(no_fs):
                fs_name = all_res.keys()[ifs]
                # writer.write(',' + fs_name + '_acc' + ',' + fs_name + '_pre' + ',' + fs_name + '_rec' + ',' + fs_name + '_f1')
                writer.write(',' + fs_name + '_f1_avg' + ',' + fs_name + '_f1_std')
            writer.write('\n')
        for num_feat in sorted(all_res[fs_alg].keys()):
            if num_feat not in result_str:
                result_str[num_feat] = dict()
            for clf_name in sorted(all_res[fs_alg][num_feat].keys()):
                scores = all_res[fs_alg][num_feat][clf_name]
                if clf_name not in result_str[num_feat]:
                    result_str[num_feat][clf_name] = ''
                # for score in scores:
                #     result_str[num_feat][clf_name] += ',' + "{:.2f}".format(score*100)
                result_str[num_feat][clf_name] += ',' + "{:.2f}".format(scores[0][-1]*100) + ",{:.2f}".format(scores[1][-1]*100)


    for num_feat in sorted(result_str.keys()):
        for clf_name in sorted(result_str[num_feat].keys()):
            writer.write(dataset + ',' + str(num_feat) + ',' + clf_name + result_str[num_feat][clf_name] + '\n')
        writer.write('\n')

    writer.close()

    # plot the results:
    for clf_name in plot_data_dict.keys():
        plot_out_path = dir_in + dataset + '_' + clf_name + '.png'
        plt.figure()
        if clf_name != 'reconstruction':
            x_line = plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][:,0]
            y_line = [plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][0,1]] * len(x_line)
            plt.plot(x_line, y_line, color='red', linestyle='--', label='baseline')
        for fs_alg in plot_data_dict[clf_name].keys():
            if fs_alg not in ['MLP', 'boruta', 'relieff', 'enet', 'lasso']:
                continue
            plot_data = plot_data_dict[clf_name][fs_alg]
            start_idx = 1
            if plot_data[1,0] == 0:
                start_idx = 2
            # plt.errorbar(plot_data[:,0], plot_data[:,1], yerr=plot_data[:,2], label=fs_alg)
            plt.plot(plot_data[start_idx:,0], plot_data[start_idx:,1], label=fs_alg)
        plt.xticks([0,5,10,15,20])
        plt.ylabel('F1 score')
        plt.xlabel('% of features')
        plt.legend(fontsize='small',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(plot_out_path,bbox_inches='tight',dpi=600)
        plt.close()

    for clf_name in plot_data_dict.keys():
        plot_out_path = dir_in + dataset + '_' + clf_name + '_DL_based.png'
        plt.figure()
        if clf_name != 'reconstruction':
            x_line = plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][:,0]
            y_line = [plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][0,1]] * len(x_line)
            plt.plot(x_line, y_line, color='red', linestyle='--', label='baseline')
        for fs_alg in plot_data_dict[clf_name].keys():
            if fs_alg not in ['MLP', 'MCFS', 'TSFS']:
                continue
            plot_data = plot_data_dict[clf_name][fs_alg]
            start_idx = 1
            if plot_data[1,0] == 0:
                start_idx = 2
            # plt.errorbar(plot_data[:,0], plot_data[:,1], yerr=plot_data[:,2], label=fs_alg)
            plt.plot(plot_data[start_idx:,0], plot_data[start_idx:,1], label=fs_alg)
        plt.xticks([0,5,10,15,20])
        plt.ylabel('F1 score')
        plt.xlabel('% of features')
        plt.legend()
        plt.savefig(plot_out_path,dpi=600)
        plt.close()

    for clf_name in plot_data_dict.keys():
        plot_out_path = dir_in + dataset + '_' + clf_name + '_elastic_lasso.png'
        plt.figure()
        if clf_name != 'reconstruction':
            x_line = plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][:,0]
            y_line = [plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][0,1]] * len(x_line)
            plt.plot(x_line, y_line, color='red', linestyle='--', label='baseline')
        for fs_alg in plot_data_dict[clf_name].keys():
            if fs_alg not in ['enet', 'lasso', 'hsic_lasso']:
                continue
            plot_data = plot_data_dict[clf_name][fs_alg]
            start_idx = 1
            if plot_data[1,0] == 0:
                start_idx = 2
            # plt.errorbar(plot_data[:,0], plot_data[:,1], yerr=plot_data[:,2], label=fs_alg)
            plt.plot(plot_data[start_idx:,0], plot_data[start_idx:,1], label=fs_alg)
        plt.xticks([0,5,10,15,20])
        plt.ylabel('F1 score')
        plt.xlabel('% of features')
        plt.legend()
        plt.savefig(plot_out_path,dpi=600)
        plt.close()

    for clf_name in plot_data_dict.keys():
        plot_out_path = dir_in + dataset + '_' + clf_name + '_others.png'
        plt.figure()
        if clf_name != 'reconstruction':
            x_line = plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][:,0]
            y_line = [plot_data_dict[clf_name][plot_data_dict[clf_name].keys()[0]][0,1]] * len(x_line)
            plt.plot(x_line, y_line, color='red', linestyle='--', label='baseline')
        for fs_alg in plot_data_dict[clf_name].keys():
            if fs_alg not in ['relieff', 'boruta', 'svm_rfe']:
                continue
            plot_data = plot_data_dict[clf_name][fs_alg]
            start_idx = 1
            if plot_data[1,0] == 0:
                start_idx = 2
            # plt.errorbar(plot_data[:,0], plot_data[:,1], yerr=plot_data[:,2], label=fs_alg)
            plt.plot(plot_data[start_idx:,0], plot_data[start_idx:,1], label=fs_alg)
        plt.ylabel('F1 score')
        plt.xlabel('% of features')
        plt.xticks([0,5,10,15,20])
        plt.legend()
        plt.savefig(plot_out_path,dpi=600)
        plt.close()





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
join_all(result_dir)