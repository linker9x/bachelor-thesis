import os
import re


# remove result for other classifiers
def update_file(in_path, out_path, fold_dict):
    with open(in_path, 'r') as f:
        lines = f.readlines()
    writer = open(out_path, 'w')

    writer.write(lines[0] + '\n')
    alg_name = ''

    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line == '':
            writer.write('\n')
            continue
        line = re.sub('^CAE\_U([0-9]*)', 'CAE', line)
        toks = line.split(',')
        num_feat = int(float(toks[2]))
        if toks[3] not in ['Neural Net', 'Linear SVM', 'reconstruction']:
            continue
        if num_feat != 0:
            alg_name = toks[0]
            writer.write(line + '\n')
            continue
        writer.write(toks[0] + ',' + toks[1] + ',' + toks[2] + ',' + toks[3] + ',' + ','.join(fold_dict[toks[1]][toks[3]]) + '\n')

    #if in_path.endswith('boruta.csv'):# comment for Lisa result, uncomment for other result
    for fold_idx in fold_dict.keys():
        for alg in fold_dict[fold_idx].keys():
            writer.write(alg_name + ',' + str(fold_idx) + ',0,' + alg + ',' + ','.join(fold_dict[fold_idx][alg]) + '\n')

    writer.close()

def standardize_dir(dir):
    dir = dir.replace('\\', '/')
    if not dir.endswith('/'):
        dir += '/'
    return dir

def read_starting_dict(dir):
    file_path = dir + '/' + 'MLP.csv'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    fold_dict = dict()
    for line in lines[1:]:
        line = line.strip()
        if line == '':
            continue
        toks = line.split(',')
        if toks[3] not in ['Neural Net', 'Linear SVM', 'reconstruction']:
            continue
        if toks[1] not in fold_dict:
            fold_dict[toks[1]] = dict()
        fold_dict[toks[1]][toks[3]] = toks[4:]

    return fold_dict


def update_dir(in_dir, starting_points_dir, out_dir):
    files = os.listdir(in_dir)
    in_dir = standardize_dir(in_dir)
    out_dir = standardize_dir(out_dir)
    starting_points_dir = standardize_dir(starting_points_dir)

    print('files: ', files)
    for file in files:
        if not os.path.isdir(in_dir + file) or not os.path.isdir(starting_points_dir + file):
            continue

        fold_dict = read_starting_dict(starting_points_dir + file)
        if not os.path.exists(out_dir + file):
            os.mkdir(out_dir + file)
        sub_dir = in_dir + file + '/'
        for sub_file in os.listdir(sub_dir):
            if not sub_file.endswith('.csv'):
                continue
            update_file(sub_dir + sub_file, out_dir + file + '/' + sub_file, fold_dict)

in_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/result_Jul9/'
starting_point_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/starting_points/'
out_dir = '/home/ngandong/Desktop/Code/feature_selection/bachelor-thesis-fs/data/standardize/subset_June30/result_Jul9_update_starting_points/'
in_dir = in_dir.replace('result_Jul9/', 'Lisa_Jul9/results/small/')
out_dir = in_dir.replace('/results/small/', '/results_update_start_point/')
update_dir(in_dir, starting_point_dir, out_dir)