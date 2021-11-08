import numpy as np
import pandas as pd
from sklearn.svm import SVC

import pickle
import os
import random

def get_predictions(df_time_1, df_time_2):
    np.random.seed(42)
    random.seed(42)
    
    classifier_time_1 = SVC(gamma='auto')
    classifier_time_2 = SVC(gamma='auto')

    classifier_time_1.fit(df_time_1[['a_', 'b_', 'c_']], df_time_1['category'])
    classifier_time_2.fit(df_time_2[['b_', 'c_', 'd_']], df_time_2['category'])

    predicted_labels_t1 = classifier_time_1.predict(df_time_1[['a_', 'b_', 'c_']])
    predicted_labels_t2 = classifier_time_2.predict(df_time_2[['b_', 'c_', 'd_']])

    return predicted_labels_t1, predicted_labels_t2


def setup_synthetic_text():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_time_1 = pd.read_csv(f'{dir_path}/test_data/df_time_1.csv')
    df_time_2 = pd.read_csv(f'{dir_path}/test_data/df_time_2.csv')

    features_t1 = ['a_', 'b_', 'c_']
    features_t2 = ['b_', 'c_', 'd_']

    # Get model and predictions
    predicted_labels_t1, predicted_labels_t2 = get_predictions(df_time_1, df_time_2)

    for col in features_t1:
        df_time_1[col] = [col if x == 1 else '' for x in df_time_1[col]]
    for col in features_t2:
        df_time_2[col] = [col if x == 1 else '' for x in df_time_2[col]]

    df_time_1['corpus'] = df_time_1['a_'] + ' ' + df_time_1['b_'] + ' ' + df_time_1['c_']
    df_time_2['corpus'] = df_time_2['b_'] + ' ' + df_time_2['c_'] + ' ' + df_time_2['d_']

    X_t1, Y_t1 = df_time_1['corpus'], df_time_1['category']
    X_t2, Y_t2 = df_time_2['corpus'], df_time_2['category']
    return (X_t1, Y_t1, predicted_labels_t1,
            X_t2, Y_t2, predicted_labels_t2)


def setup_synthetic_tabular():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_time_1 = pd.read_csv(f'{dir_path}/test_data/df_time_1.csv')
    df_time_2 = pd.read_csv(f'{dir_path}/test_data/df_time_2.csv')
    features_t1 = ['a_', 'b_', 'c_']
    features_t2 = ['b_', 'c_', 'd_']

    # Get model and predictions
    predicted_labels_t1, predicted_labels_t2 = get_predictions(df_time_1, df_time_2)

    X_t1, Y_t1 = df_time_1[features_t1], df_time_1['category']
    X_t2, Y_t2 = df_time_2[features_t2], df_time_2['category']
    return (X_t1, Y_t1, predicted_labels_t1,
            X_t2, Y_t2, predicted_labels_t2)
