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

        clf_name = tokens[2].replace('Neural Net', 'NeuralNet').replace('Linear SVM', 'LinearSVM')
        itok = 3
        while itok < len(tokens):
            iname = (itok - 3) / 2
            if iname == len(fs_alg_names):
                print(csv_path, i)
            fs_name = fs_alg_names[iname]
            if clf_name not in sum_dict:
                sum_dict[clf_name] = dict()

            if fs_name not in sum_dict[clf_name]:
                sum_dict[clf_name][fs_name] = dict()
            if num_feat != 0:
                if clf_name == 'reconstruction':
                    sum_dict[clf_name][fs_name][num_feat] = round(float(tokens[itok])/100,2)
                else:
                    sum_dict[clf_name][fs_name][num_feat] = round(float(tokens[itok]),2)
            elif clf_name not in baseline_dict:
                baseline_dict[clf_name] = round(float(tokens[itok]),2)
            if clf_name == 'reconstruction':
                if fs_name not in reconstruction_dict:
                    reconstruction_dict[fs_name] = dict()
                if num_feat != 0:
                 reconstruction_dict[fs_name][num_feat] = round(float(tokens[itok]),2)
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

    for clf_name in sum_dict.keys():
        fs_dict = sum_dict[clf_name]
        writer.write(clf_name + "\n")
        plot_data = list()
        plot_cols = [str(ifeat) for ifeat in range(1,21)]
        indexes = []
        for fs_name in sorted(fs_dict.keys()):
            indexes.append(fs_name.replace('TSFS_my_autoencoder', 'TSFS').replace('TSFS_MCFS', 'MCFS').replace('elastic_net', 'enet'))
            val_list = list()
            writer.write(fs_name + ',' + str(baseline_dict[clf_name]) + ',')

            for ifeat in sorted(fs_dict[fs_name].keys()):
                val_list.append(fs_dict[fs_name][ifeat])
            if len(val_list) < len(plot_cols):
                for itmp in range(len(val_list), len(plot_cols)):
                    val_list.append(val_list[-1])

            plot_data.append(val_list)

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
                if i_above == -1 and score >= baseline_dict[clf_name]:
                    i_above = i + 1
                if i_nearly == -1 and score <= baseline_dict[clf_name] and abs(score - baseline_dict[clf_name]) < 1:
                    i_nearly = i + 1
            writer.write(str(i_above) + ',' + str(i_nearly) + '\n')

            if max_f1 < max:
                max_entry = fs_name + ':' + clf_name + ':' + str(max_idx)
                max_f1 = max
            elif max_f1 == max:
                max_entry += ', ' + fs_name + ':' + clf_name + ':' + str(max_idx)
            if largest_mean < mean:
                largest_mean = mean
                largest_mean_entry = fs_name + ':' + clf_name
            if smallest_std > std:
                smallest_std = std
                smallest_std_entry = fs_name + ':' + clf_name
            if i_above != -1 and (earlest_above == -1 or earlest_above >= i_above):
                if earlest_above > i_above:
                    earlest_above_entry = fs_name + ':' + clf_name
                    earlest_above = i_above
                elif earlest_above == i_above:
                    earlest_above_entry += ', ' + fs_name + ':' + clf_name
                if earlest_nearly == -1:
                    earlest_above = i_above
            if i_nearly != -1 and (earlest_nearly == -1 or earlest_nearly >= i_nearly):
                if earlest_nearly > i_nearly:
                    earlest_nearly_entry = fs_name + ':' + clf_name
                    earlest_nearly = i_nearly
                elif earlest_nearly == i_nearly:
                    earlest_nearly_entry += ', ' + fs_name + ':' + clf_name
                if earlest_nearly == -1:
                    earlest_nearly = i_nearly

            stat_list= [baseline_dict[clf_name],max,min,median,mean,std]
            i_stats = [max_idx,i_above if i_above != -1 else 0,i_nearly if i_nearly != -1 else 0]

        writer.write('\n\n')
        tmp =np.asmatrix(plot_data)
        plot_df = pd.DataFrame(tmp, index=indexes, columns=plot_cols)
        plot_df = plot_df.T
        plot_out_path = csv_path.replace('_sum_join.csv','/' + dataset + '_' + clf_name + '_rplot.png')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_df.boxplot(figsize=(19,10),
                        whis=[5,95], return_type='axes')#grid=False, rot=45,
        if clf_name != 'reconstruction':
            x = range(len(indexes)+2)
            y = [baseline_dict[clf_name]] * (len(indexes) + 2)
            plt.plot(x, y, color='red', linestyle='--')
            # plt.ylabel('F1 score', fontsize =18)
        # else:
            # plt.ylabel('Mean Square Error', fontsize =18)
        # plt.xlabel('Feature selection methods', fontsize =18)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        # print(dataset, plot_df.shape)
        # dump = np.zeros((len(plot_cols), len(indexes) + 1))
        # print(dump.shape, np.reshape(plot_df[['enet']].values, (len(plot_cols))).shape)
        # dump[:, len(indexes)] = plot_df[['enet']].values[:,0]
        # more_df = pd.DataFrame(dump, columns=indexes.append('dump'))
        # print(more_df.shape)
        # more_df.boxplot(fontsize ='small', figsize=(19,10),
        #                  whis=[5,95], return_type='axes')#grid=False, rot=45,
        # # plt.legend(fontsize='small',loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(plot_out_path,bbox_inches='tight',dpi=600)
        plt.close()


    writer.write('\n\nIN SUM:\n')
    writer.write('max_f1 value:, ' + "{:.2f}".format(max_f1) + ', acquire by:, ' + max_entry + '\n')
    writer.write('largest mean value:, ' + "{:.2f}".format(largest_mean) + ' ,acquire by:, ' + largest_mean_entry + '\n')
    writer.write('min std value:, ' + "{:.2f}".format(smallest_std) + ', acquire by: ,' + smallest_std_entry + '\n')
    writer.write('earlest above value: ,' + str(earlest_above) + ' ,acquire by: ,' + earlest_above_entry + '\n')
    writer.write('earlest nearly value:, ' + str(earlest_nearly) + ', acquire by:, ' + earlest_nearly_entry + '\n')


    writer.write('\n\nReconstruction error:\n')
    writer.write(',max,min,median,mean,std,min_idx\n')

    indexes = []
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
        indexes.append(fs_clf_name.replace('TSFS_my_autoencoder', 'TSFS').replace('TSFS_MCFS', 'MCFS'))


    # svm_resconstruction_df = pd.DataFrame(np.asmatrix(res_data), index=indexes, columns=res_col)
    # plot_out_path = csv_path.replace('.csv','_reconstruction.png')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # svm_resconstruction_df.plot(kind='bar')#,stacked=True
    # plt.legend()
    # plt.savefig(plot_out_path,bbox_inches='tight',dpi=100)
    # plt.close()


    writer.close()
    return {dataset: res_dict}

def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir


def gen_rplot_all(dir):
    files = os.listdir(dir)
    dir = standardize_dir(dir)
    for file in files:
        if not file.endswith('_sum_join.csv'):
            continue

        read_sum_file(dir + file, dir + file.replace('_sum_join.csv', '_stats.csv'))
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

# read_sum_file(result_dir + 'Prostate_GE_sum_result.csv', result_dir + 'Prostate_GE_stats.csv')
gen_rplot_all(result_dir)

file = result_dir + 'Data_Cortex_Nuclear_sum_join.csv'
# read_sum_file(file, file.replace('_sum_join.csv', '_stats.csv'))