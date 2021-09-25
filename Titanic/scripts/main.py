'''
If you find this useful, please give a thumbs up!

Thanks!
- Claire & Alhan
'''

# External libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import scipy.stats as stats
import math
import time
import traceback
import warnings
import os

# Options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
warnings.filterwarnings(action="ignore")


class Explore:

    def get_dtype(cls, include_type=[], exclude_type=[]):
        df = cls.get_df('train')
        df.drop(columns=[cls.target_col], inplace=True)
        return df.select_dtypes(include=include_type, exclude=exclude_type)

    def get_non_numeric(cls):
        return cls.get_dtype(exclude_type=['float64', 'int', 'float32'])

    def get_numeric(cls):
        return cls.get_dtype(exclude_type=['object', 'category'])

    def get_categorical(cls, as_df=False):
        return cls.get_dtype(include_type=['object'])

    def get_correlations(cls, method='spearman'):
        df = cls.get_df('train')
        corr_mat = df.corr(method=method)
        corr_mat.sort_values(cls.target_col, inplace=True)
        corr_mat.drop(cls.target_col, inplace=True)
        return corr_mat[[cls.target_col]]

    def get_skewed_features(cls, df, features, skew_threshold=0.4):
        feat_skew = pd.DataFrame(
                    {'skew': df[features].apply(lambda x: stats.skew(x))})
        skewed = feat_skew[abs(feat_skew['skew']) > skew_threshold].index
        return skewed.values

    def show_boxplot(cls, x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x = plt.xticks(rotation=90)

    def plot_categorical(cls, df, cols):
        target = cls.target_col
        categorical = pd.melt(df, id_vars=[target],
                              value_vars=cols)
        grouped = categorical.groupby(['value', 'variable'],
                                      as_index=False)[target]\
            .mean().rename(columns={target: target + '_Mean'})
        categorical = pd.merge(categorical, grouped, how='left',
                               on=['variable', 'value'])\
            .sort_values(target + '_Mean')
        facet_grid = sns.FacetGrid(categorical, col="variable",
                                   col_wrap=3, size=5,
                                   sharex=False, sharey=False,)
        facet_grid = facet_grid.map(cls.show_boxplot, "value", target)
        plt.savefig('boxplots.png')


class Clean:

    def keep_only_keep(cls, df):
        to_drop = set(df.columns.values) - set(cls.keep)
        if df.name == 'train':
            to_drop = to_drop - set([cls.target_col])
        to_drop = list(to_drop)
        df.drop(to_drop, axis=1, inplace=True)
        return df

    def remove_outliers(cls, df):
        if df.name == 'train':
            # GrLivArea (1299 & 524)
            # df.drop(df[(df['GrLivArea'] > 4000) &
            #         (df[cls.target_col] < 300000)].index,
            #         inplace=True)
            pass
        return df

    def fill_by_type(cls, x, col):
        if pd.isna(x):
            if col.dtype == 'object':
                return 0
            return 0
        return x

    def fill_na(cls, df):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: cls.fill_by_type(x, df[col]))
        return df

    def get_encoding_lookup(cls, cols):
        df = cls.get_df('train')
        target = cls.target_col
        suffix = '_E'
        result = pd.DataFrame()
        for cat_feat in cols:
            cat_feat_target = df[[cat_feat, target]].groupby(cat_feat)
            cat_feat_encoded_name = cat_feat + suffix
            order = pd.DataFrame()
            order['val'] = df[cat_feat].unique()
            order.index = order.val
            order.drop(columns=['val'], inplace=True)
            order[target + '_mean'] = cat_feat_target[[target]].median()
            order['feature'] = cat_feat
            order['encoded_name'] = cat_feat_encoded_name
            order = order.sort_values(target + '_mean')
            order['num_val'] = range(1, len(order)+1)
            result = result.append(order)
        result.reset_index(inplace=True)
        return result

    def get_scaled_categorical(cls, encoding_lookup):
        scaled = encoding_lookup.copy()
        target = cls.target_col
        for feature in scaled['feature'].unique():
            values = scaled[scaled['feature'] == feature]['num_val'].values
            medians = scaled[
                    scaled['feature'] == feature][target + '_mean'].values
            for median in medians:
                scaled_value = ((values.min() + 1) *
                                (median / medians.min()))-1
                scaled.loc[(scaled['feature'] == feature) &
                           (scaled[target + '_mean'] == median),
                           'num_val'] = scaled_value
        return scaled

    def encode_with_lookup(cls, df, encoding_lookup):
        for encoded_index, encoded_row in encoding_lookup.iterrows():
            feature = encoded_row['feature']
            encoded_name = encoded_row['encoded_name']
            value = encoded_row['val']
            encoded_value = encoded_row['num_val']
            df.loc[df[feature] == value, encoded_name] = encoded_value
        return df

    def encode_onehot(cls, df, cols):
        df = pd.concat([df, pd.get_dummies(df[cols], drop_first=True)], axis=1)
        return df

    def encode_categorical(cls, df, cols=[], method='one_hot'):
        if len(cols) == 0:
            cols = cls.get_categorical().columns.values
        if method == 'target_mean':
            encoding_lookup = cls.get_encoding_lookup(cols)
            encoding_lookup = cls.get_scaled_categorical(encoding_lookup)
            df = cls.encode_with_lookup(df, encoding_lookup)
        if method == 'one_hot':
            if len(set(cols) - set(cls.get_dtype(include_type=['object'])
                   .columns.values)) > 0:
                    for col in cols:
                        df[col] = df[col].apply(lambda x: str(x))
            df = cls.encode_onehot(df, cols)
        df.drop(cols, axis=1, inplace=True)
        return df

    def fix_zero_infinity(cls, x):
        if (x == 0) or math.isinf(x):
            return 0
        return x

    def normalize_features(cls, df, cols=[]):
        if len(cols) == 0:
            cols = cls.get_numeric().columns.values
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x:
                                        np.log1p(x).astype('float64'))
                df[col] = df[col].apply(lambda x: cls.fix_zero_infinity(x))
        return df

    def scale_quant_features(cls, df, cols):
        scaler = StandardScaler()
        scaler.fit(df[cols])
        scaled = scaler.transform(df[cols])
        for i, col in enumerate(cols):
            df[col] = scaled[:, i]
        return df

    def drop_ignore(cls, df):
        for col in cls.ignore:
            try:
                df.drop(col, axis=1, inplace=True)
            except Exception:
                pass
        return df

    def drop_low_corr(cls, df, threshold=0.12):
        to_drop = pd.DataFrame(columns=['drop'])
        corr_mat = cls.get_correlations()
        target = cls.target_col
        to_drop['drop'] = corr_mat[(abs(corr_mat[target]) <= threshold)].index
        df.drop(to_drop['drop'], axis=1, inplace=True)
        return df


class Engineer:

    def encode_features(cls, df, cols):
        train, test = cls.get_dfs()
        df_combined = pd.concat([train[cols], test[cols]])
        train.drop(cls.target_col, axis=1, inplace=True)
        for feature in cols:
            le = LabelEncoder()
            df_combined[feature] = df_combined[feature].apply(lambda x: str(x))
            df[feature] = df[feature].apply(lambda x: str(x))
            le = le.fit(df_combined[feature])
            df[feature] = le.transform(df[feature])
        return df

    def simplify_feature(cls, df):
        # Example setup for such a function:
        df['Age'] = df['Age'].fillna(-0.5)
        bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
        group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student',
                       'Young Adult', 'Adult', 'Senior']
        categories = pd.cut(df['Age'], bins, labels=group_names)
        df['Age'] = categories
        return df

    def sum_features(cls, df, col_sum):
        for col_set in col_sum:
            f_name = '__'.join(col_set[:])
            df[f_name] = df[[*col_set]].sum(axis=1)
            df.drop(col_set, axis=1, inplace=True)
        return df

    def combine_features(cls, row, col_set):
        result = ''
        for col in col_set:
            if result != '':
                result += '_'
            result += str(row[col])
        return result

    def combine(cls, df, col_sets):
        for col_set in col_sets:
            f_name = '__'.join(col_set[:])
            df[f_name] = df.apply(lambda x: cls.combine_features(x, col_set),
                                  axis=1)
            df.drop(col_set, axis=1, inplace=True)
        return df

    def multiply_features(cls, df, feature_sets):
        for feature_set in feature_sets:
            # multipled_name = '_x_'.join(feature_set[:])
            # df.drop(feature_set, axis=1, inplace=True)
            pass
        return df


class Model:

    def fix_shape(cls, df):
        df_name = df.name
        if df_name == 'train':
            cols_to_add = set(cls.get_df('test').columns.values) -\
                          set(df.drop(cls.target_col, axis=1).columns.values)
        if df_name == 'test':
            cols_to_add = set(cls.get_df('train').drop(cls.target_col, axis=1)
                              .columns.values) - set(df.columns.values)
        cols_to_add = np.array(list(cols_to_add))
        cols_to_add = np.append(cols_to_add, df.columns.values)
        df = df.reindex(columns=cols_to_add, fill_value=0)
        df.name = df_name
        return df

    def cross_validate(cls, model, parameters):
        train, test = cls.get_dfs()
        # TODO: check if there are lists in parameters to run gridsearch
        if len(train.drop(cls.target_col,
               axis=1).columns) != len(test.columns):
            cls.mutate(cls.fix_shape)
            train = cls.get_df('train')
        scores = np.array([])
        skf = StratifiedKFold(n_splits=10, random_state=None)
        X = train.drop(columns=[cls.target_col])
        y = train[cls.target_col]
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            cv_model = model(**parameters)
            cv_model.fit(X_train, y_train)
            X_predictions = cv_model.predict(X_test)
            score = accuracy_score(y_test, X_predictions)
            scores = np.append(scores, score)
        score = np.round(scores.mean(), decimals=5)
        return score

    def fit(cls, model, parameters):
        train = cls.get_df('train')
        X = train.drop(columns=[cls.target_col])
        y = train[cls.target_col]
        model = model(**parameters)
        model.fit(X, y)
        return model

    def predict(cls, model):
        test = cls.get_df('test')
        predictions = model.predict(test)
        return predictions

    def save_predictions(cls, predictions, score=0, id_col=False):
        now = str(time.time()).split('.')[0]
        df = cls.get_df('test', False, True)
        target = cls.target_col
        df[target] = predictions
        if not id_col:
            id_col = df.columns[0]
        if not os.path.exists(path + '/output'):
            os.makedirs(path + '/output')
        if os.path.exists(path + '/output'):
            df[[id_col,
                target]].to_csv(path + '/output/submit__' +
                                str(int(score * 100000))
                                + '__' + now + '.csv', index=False)
        df[[id_col, target]].to_csv('submission.csv', index=False)


class Data(Explore, Clean, Engineer, Model):

    def __init__(self, train_csv, test_csv, target='',
                 ignore=[], keep=[], col_sum=[]):
        '''Create pandas DataFrame objects for train and test data.

        Positional arguments:
        train_csv -- relative path to training data in csv format.
        test_csv -- relative path to test data in csv format.

        Keyword arguments:
        target -- target feature column name in training data.
        ignore -- columns names in list to ignore during analyses.
        '''
        self.__train = pd.read_csv(train_csv)
        self.__test = pd.read_csv(test_csv)
        self.__train.name, self.__test.name = self.get_df_names()
        self.target_col = target
        self.ignore = ignore
        self.keep = keep
        self.col_sum = col_sum
        self.__original = False
        self.__log = False
        self.check_in()
        self.debug = False

    def __str__(cls):
        train_columns = 'Train: \n"' + '", "'.join(cls.__train.head(2)) + '"\n'
        test_columns = 'Test: \n"' + '", "'.join(cls.__test.head(2)) + '"\n'
        return train_columns + test_columns

    def get_df_names(cls):
        return ('train', 'test')

    def get_dfs(cls, ignore=False, originals=False, keep=False):
        train, test = (cls.__train.copy(),
                       cls.__test.copy())
        if originals:
            train, test = (cls.__original)
        if ignore:
            train, test = (train.drop(columns=cls.ignore),
                           test.drop(columns=cls.ignore))
        if keep:
            train, test = (train[cls.keep],
                           test[cls.keep])
        train.name, test.name = cls.get_df_names()
        return (train, test)

    def get_df(cls, name, ignore=False, original=False, keep=False):
        train, test = cls.get_dfs(ignore, original, keep)
        if name == 'train':
            return train
        if name == 'test':
            return test

    def log(cls, entry=False, status=False):
        if cls.__log is False:
            cls.__log = pd.DataFrame(columns=['entry', 'status'])
        log_entry = pd.DataFrame({'entry': entry, 'status': status}, index=[0])
        cls.__log = cls.__log.append(log_entry, ignore_index=True)
        if status == 'Fail':
            cls.rollback()
        else:
            cls.check_out()
            if cls.debug:
                cls.print_log()

    def print_log(cls):
        print(cls.__log)

    def check_in(cls):
        cls.__current = cls.get_dfs()
        if cls.__original is False:
            cls.__original = cls.__current

    def check_out(cls):
        cls.__previous = cls.__current
        cls.__train.name, cls.__test.name = cls.get_df_names()

    def rollback(cls):
        try:
            cls.__train, cls.__test = cls.__previous
            status = 'Success - To Previous'
        except Exception:
            cls.__train, cls.__test = cls.__original
            status = 'Success - To Original'
        cls.log('rollback', status)

    def reset(cls):
        cls.__train, cls.__test = cls.__original
        cls.log('reset', 'Success')

    def update_dfs(cls, train, test):
        train.name, test.name = cls.get_df_names()
        cls.__train = train
        cls.__test = test

    def mutate(cls, mutation, *args):
        '''Make changes to both train and test DataFrames.
        Positional arguments:
        mutation -- function to pass both train and test DataFrames to.
        *args -- arguments to pass to the function, following each DataFrame.

        Example usage:
        def multiply_column_values(df, col_name, times=10):
            #do magic...

        Data.mutate(multiply_column_values, 'Id', 2)
        '''
        cls.check_in()
        try:
            train = mutation(cls.get_df('train'), *args)
            test = mutation(cls.get_df('test'), *args)
            cls.update_dfs(train, test)
            status = 'Success'
        except Exception:
            print(traceback.print_exc())
            status = 'Fail'
        cls.log(mutation.__name__, status)


def run(d, model, parameters):
    mutate = d.mutate
    # mutate(d.sum_features, d.col_sum)
    # mutate(d.combine, d.col_sum)
    # mutate(d.keep_only_keep)
    mutate(d.fill_na)
    # mutate(d.encode_features)
    mutate(d.drop_ignore)
    score = d.cross_validate(model, parameters)
    print(score)
    model = d.fit(model, parameters)
    predictions = d.predict(model)
    d.print_log()
    train = d.get_df('train')
    print(train.columns.values)
    return (predictions, score)


model = LogisticRegression
parameters = {}
cols_to_ignore = ['PassengerId']
id_col = 'PassengerId'
path = '.'
if os.getcwd().split('/')[1] == 'kaggle':
    path = '..'
d = Data(path + '/input/train.csv',
         path + '/input/test.csv',
         'Survived',
         ignore=cols_to_ignore)
predictions, score = run(d, model, parameters)
d.save_predictions(predictions, score, id_col)
