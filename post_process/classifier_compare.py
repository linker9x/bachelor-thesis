import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def gen_plots(csv_path, dir_out):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    head_toks = lines[0].strip().split(',')
    fs_cols = list()
    for i in range(3, len(head_toks)):
        if i % 2 == 0:
            continue
        fs_col = head_toks[i].replace('_f1_avg', '').replace('TSFS_my_autoencoder', 'TSFS').replace('TSFS_MCFS', 'MCFS')
        fs_cols.append(fs_col)

    feat_cols = []
    feat_dict = dict()
    fs_dict = dict()
    dataset = ''
    clf_name_dict = dict()
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line == '':
            continue
        toks = line.split(',')

        if toks[1].endswith('.0'):
            continue
        # Feed data into feature dict (feat_dict)
        ifeat = int(toks[1])
        clf_name = toks[2]
        if ifeat == 0: # Add headers
            dataset = toks[0]
            feat_cols.append(clf_name)
            clf_name_dict[clf_name] = len(clf_name_dict)

        if ifeat not in feat_dict:
            feat_dict[ifeat] = list()
        feat_vals = list()
        for j in range(3, len(toks)):
            if j % 2 == 0:
                continue
            feat_vals.append(float(toks[j]))
        feat_dict[ifeat].append(feat_vals)
        # end feat_dict feed

        # Feed data in to feature selection dict (fs_dict)
        for j in range(3, len(toks)):
            if j % 2 == 0:
                continue
            fs_idx = (j-3)/2
            fs_name = fs_cols[fs_idx]
            if fs_name not in fs_dict:
                fs_dict[fs_name] = list()
            if len(fs_dict[fs_name]) == clf_name_dict[clf_name]:
                fs_dict[fs_name].append(list())
            fs_dict[fs_name][clf_name_dict[clf_name]].append(float(toks[j]))

    # draw plots according to number of selected features
    # for ifeat in feat_dict.keys():
    #     plot_out_path = dir_out + dataset + '_' + str(ifeat) + '.png'
    #     df = pd.DataFrame(np.asmatrix(feat_dict[ifeat]), index = feat_cols)
    #     df = df.T
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     df.boxplot(fontsize ='small', figsize=(19,10),
    #                whis=[5,95], return_type='axes')#grid=False, rot=45,
    #     plt.ylabel('Complexity score')
    #     plt.xlabel('Fold')
    #     plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #     plt.tight_layout()
    #     plt.savefig(plot_out_path,bbox_inches='tight',dpi=600)
    #     plt.close()
    #
    # draw plots according to number of feature selection algorithm
    for fs in fs_dict.keys():
        plot_out_path = dir_out + dataset + '_' + fs + '.png'
        # print(len(feat_cols), feat_cols)
        # print(len(fs_dict[fs]))
        df = pd.DataFrame(np.asmatrix(fs_dict[fs]), index=feat_cols)
        df = df.T

        fig = plt.figure()
        ax = fig.add_subplot(111)
        df.boxplot(fontsize ='small', figsize=(19,10),
                   whis=[5,95], return_type='axes')#grid=False, rot=45,
        plt.ylabel('Complexity score')
        plt.xlabel(dataset)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()
        plt.savefig(plot_out_path,bbox_inches='tight',dpi=600)
        plt.close()
    plot_data = list()
    for ifeat in sorted(feat_dict.keys()):
        vals = list()
        for tmplist in feat_dict[ifeat]:
            arr = np.asmatrix(tmplist)
            vals.append(np.mean(arr))
        plot_data.append(vals)
    plot_out_path = dir_out + dataset + '.png'
    print(len(plot_data), len(feat_cols))
    matrix = np.asmatrix(plot_data)
    df = pd.DataFrame(matrix[:, :5], columns=feat_cols[:len(feat_cols)-1])
    # df = df.T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.boxplot(fontsize ='small', figsize=(19,10),
                   whis=[5,95], return_type='axes')#grid=False, rot=45,
    plt.ylabel('Complexity score')
    plt.xlabel(dataset)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(plot_out_path,bbox_inches='tight',dpi=600)
    plt.close()




def gen_all(dir, dir_out):
    files = os.listdir(dir)
    for file in files:
        if not file.endswith('_sum_result.csv'):
            continue
        print('Processing ', file)
        gen_plots(dir + file, dir_out)

dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/compare_classifier/'
dir_out = dir + 'plots/'
chin = dir + 'chin_sum_result.csv'
tox171 = dir + 'TOX_171_sum_result.csv'
# gen_plots(tox171, dir_out)
gen_all(dir, dir_out)