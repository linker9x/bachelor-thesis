import pandas as pd
import re
from pandas.api.types import is_numeric_dtype
from numpy.random import RandomState
from scipy.stats import zscore
from post_process.ba_postprocessor import

def preprocess(csv_filepath, class_name, remove_cols=[], frac=0.8):
    try:
        df = pd.read_csv(csv_filepath)

        # rename the class attribute to 'class'
        if class_name not in df:
            raise ValueError("Class name invalid for %s." % csv_filepath)

        df.rename(columns={class_name: 'class'}, inplace=True)

        # remove columns
        if remove_cols:
            print('DROPPED: %s' % df.columns[remove_cols])
            df.drop(df.columns[remove_cols], axis=1, inplace=True)

        # remove the class attribute for feature preprocessing
        class_col = df.pop('class')

        # remove all cols that are not numerical
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                tmp = df.pop(col)
                print('Column %s removed. Type %s not numeric.' % (col, tmp.dtype))

        # replace missing values with column mean
        # df.fillna(df.mean(), inplace=True)
        df = df.interpolate(method='nearest', axis=0).ffill().bfill()

        # Z Normalization
        #print(df.describe())
        df = df.apply(zscore)

        # add the class attribute back to the dataframe at the end
        df['class'] = class_col
        df['class'] = pd.factorize(df['class'])[0] + 1

        # # save
        # rng = RandomState(42)
        # train = df.sample(frac=frac, random_state=rng)
        # test = df.loc[~df.index.isin(train.index)]

        # train.to_csv(re.sub('\.csv$', '', csv_filepath) + '_train.csv', index=False)
        # test.to_csv(re.sub('\.csv$', '', csv_filepath) + '_test.csv', index=False)
        df.to_csv(re.sub('\.csv$', '', csv_filepath) + '_clean.csv', index=False)

    except ValueError as ve:
        print(ve)
