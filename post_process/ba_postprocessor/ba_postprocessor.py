import pandas as pd

c_name = 'NN'

def average_results(ds_name, alg_names):
    for alg in alg_names:
        # algorithms returning subset of features
        if alg in ['EN', 'LASSO', 'SS', 'HSIC', 'CFS', 'JACKSTRAW']:
            data = pd.read_csv('./results/' + ds_name + '/' + alg + '.csv')
            data = data[['alg_name', 'per_feat', 'clf_name', 'f1']]
            data = data[~data['clf_name'].isin(['Random Forest', 'Linear SVM'])]

            # average of 5-folds, group by classifier and percent of features
            averages = data.groupby(['per_feat', 'clf_name']).mean()
            averages = averages.reset_index()
            averages.to_csv('./results/' + ds_name + '/' + alg + '_' + c_name + '_avg.csv', index=None, header=True)

        # algorithms returning all features
        else:
            data = pd.read_csv('./results/' + ds_name + '/' + alg + '.csv')
            data = data[['alg_name', 'num_feat', 'clf_name', 'f1']]
            data = data[~data['clf_name'].isin(['Random Forest', 'Linear SVM'])]

            # average of 5-folds, group by classifier and percent of features
            averages = data.groupby(['num_feat', 'clf_name']).mean()
            averages = averages.reset_index()
            averages.to_csv('./results/' + ds_name + '/' + alg + '_' + c_name + '_avg.csv', index=None, header=True)


def ranked_list(ds_name, alg_names, ascending=False):
    max_vals = pd.DataFrame(columns=['alg_name', 'feat', 'max_f1_val', 'avg_f1_val'])

    for alg in alg_names:
        results = pd.read_csv('./results/' + ds_name + '/' + alg + '_' + c_name + '_avg.csv')
        idx = results['f1'].idxmax()
        avg = results['f1'].mean()

        if alg in ['EN', 'LASSO', 'SS', 'HSIC', 'CFS', 'JACKSTRAW']:
            rank_by = 'per_feat'
        else:
            rank_by = 'num_feat'

        max_val = pd.DataFrame({'alg_name': [alg],
                                'feat': [results.iloc[idx][rank_by]],
                                'max_f1_val': [results.iloc[idx]['f1']],
								'avg_f1_val': [avg]
                                })
        max_vals = max_vals.append(max_val, ignore_index=True)

    max_vals = max_vals.sort_values(by=['max_f1_val', 'avg_f1_val'], ascending=[ascending, ascending])
    max_vals.to_csv('./results/' + ds_name + '/ranked_list_' + str(ascending) + '_' + c_name + '.csv', index=None, header=True)

