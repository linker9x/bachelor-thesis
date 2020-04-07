import numpy as np

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def get_remain(lst, intersect):
    res = [n for n in lst if n not in intersect]
    return res

def tanimoto_dist(lst1, lst2):
    len_both = len(lst1) + len(lst2)
    len_intersection = len(intersection(lst1, lst2))
    return 1 - (len_both - 2 * len_intersection) * 1.0 / (len_both - len_intersection)

def hamming_dist(lst1, lst2):
    len_both = len(lst1) + len(lst2)
    a_minus_b = get_remain(lst1, lst2)
    b_minus_a = get_remain(lst2, lst1)
    return 1 - 1.0 * (len(a_minus_b) + len(b_minus_a)) / len_both

def kuncheva_dist(lst1, lst2, n):
    t = float(len(lst1))
    intersect = intersection(lst1, lst2)
    if t== n:
        return -1
    return 1.0 * (len(intersect) - t / n) / (t - t * t / n)

def jaccard_idx(lst1, lst2):
    len_both = float(len(lst1) + len(lst2))
    len_intersection = len(intersection(lst1, lst2))
    return len_intersection / (len_both - len_intersection)


# convert sang dictionary
sim_name = ['tanimoto', 'hamming', 'kuncheva', 'jaccard']
def get_smilarity_scores(fold_lst, n_features):
    scores = {'tanimoto' : [], 'hamming' : [], 'kuncheva' : [], 'jaccard' : []}
    for i in range(0, len(fold_lst)-1):
        for j in range(i + 1, len(fold_lst)):
            lst1 = fold_lst[i]
            lst2 = fold_lst[j]
            scores['tanimoto'].append(tanimoto_dist(lst1, lst2))
            scores['hamming'].append(hamming_dist(lst1, lst2))
            scores['kuncheva'].append(kuncheva_dist(lst1, lst2, n_features))
            scores['jaccard'].append(jaccard_idx(lst1, lst2))

    res = dict()
    for key in sim_name:
        arr = np.matrix(scores[key])
        res[key] = [0, 0, np.std(arr),np.mean(arr)]
    return res

# lst1 = [15, 9, 10, 56, 23, 78, 5, 4, 9]
# lst2 = [9, 4, 5, 36, 47, 26, 10, 45, 87]
# fold_list = [lst1, lst2]
# print(hamming_dist(lst1, lst2))
# print(jaccard_idx(lst1, lst2))
# print(tanimoto_dist(lst1, lst2))
# print(get_smilarity_scores(fold_list, 10))
