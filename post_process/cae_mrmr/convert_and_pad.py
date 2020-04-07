import numpy as np
import os

def convert(csv_in, csv_out, step = 10):
    with open(csv_in, 'r') as f:
        lines = f.readlines()
    writer = open(csv_out, 'w')

    writer.write(lines[0].strip() + '\n')

    header_toks = lines[0].strip().split(',')
    itok = 3
    fs_alg_names = list()
    while itok < len(header_toks):
        fs_alg_names.append(header_toks[itok])
        itok += 1

    sum_dict = dict()
    dataset = ''
    for i in range(1, len(lines)):
        line = lines[i]
        line = line.strip()
        if line == '':
            # writer.write('\n')
            continue
        tokens = line.split(',')
        dataset = tokens[0]
        num_feat = int(tokens[1])/step
        if num_feat == 0:
            writer.write(line + '\n')
            continue

        clf_name = tokens[2]
        itok = 3
        while itok < len(tokens):
            iname = (itok - 3)
            fs_name = fs_alg_names[iname]
            if clf_name not in sum_dict:
                sum_dict[clf_name] = dict()

            if fs_name not in sum_dict[clf_name]:
                sum_dict[clf_name][fs_name] = dict()

            sum_dict[clf_name][fs_name][num_feat] = round(float(tokens[itok]),2)

            itok += 1

    resultStr = dict()
    for clf_name in sum_dict.keys():
        if clf_name not in resultStr:
            resultStr[clf_name] = dict()

        for fs_name in fs_alg_names:
            feat_dict = sum_dict[clf_name][fs_name]
            val_list = []
            for key in sorted(feat_dict.keys()):
                if key not in resultStr[clf_name]:
                    resultStr[clf_name][key] = dataset + ',' + str(key) + ',' + clf_name
                resultStr[clf_name][key] +=  ',' + str(feat_dict[key])
                val_list.append(feat_dict[key])

            num_of_pad = 20 - len(feat_dict.keys())
            if num_of_pad == 0:
                continue
            mean_val = np.mean(np.array(val_list)) # add mean value into the missing value
            std_val = np.std(np.array(val_list)) / num_of_pad
            for ikey in range(len(feat_dict.keys())+1, 21):
                if ikey not in resultStr[clf_name]:
                    resultStr[clf_name][ikey] = dataset + ',' + str(ikey) + ',' + clf_name
                pad_val = mean_val - std_val
                if ikey % 2 == 0:
                    pad_val = mean_val + std_val
                resultStr[clf_name][ikey] +=  ',' + "{:.2f}".format(pad_val)



    for clf_name in resultStr.keys():
        for ifeat in resultStr[clf_name].keys():
            writer.write(resultStr[clf_name][ifeat] + '\n')
        writer.write('\n')

    writer.close()


def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir


def convert_all(dir):
    files = os.listdir(dir)
    dir = standardize_dir(dir)
    for file in files:
        if not file.endswith('_sum_result.csv'):
            continue
        out_name = file.replace('_sum_result.csv', '_sum_padded.csv')
        if file in ['Data_Cortex_Nuclear_sum_result.csv','LSVT_sum_result.csv']:
            convert(dir + file, dir + out_name, step=5)
        else:
            convert(dir + file, dir + out_name, step=10)

dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/Lisa_Jul9/results/'
convert_all(dir)