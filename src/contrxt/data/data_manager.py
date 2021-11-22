import numpy as np
import pandas as pd
import string
from typing import Dict, List

import logging
from contrxt.util.logger import build_logger


class DataManager():

    def __init__(self, X, Y_predicted, time_label):
        self.logger = build_logger(logging.DEBUG, __name__, 'logs/trace.log')

        self.X = X
        self.Y_predicted = pd.Series([self.check_column_names(y) for y in Y_predicted])
        self.Y_predicted_binarized = {}
        self.time_label = time_label
        self.feature_names = None
        self.surrogate_train_data = {}

        try:
            if self.X.shape[1]:
                self.df = pd.DataFrame(self.X.copy())
                self.df['Y_predicted'] = self.Y_predicted.copy()
        except IndexError:
            self.df = pd.DataFrame({
                'X': self.X.copy(),
                'Y_predicted': self.Y_predicted.copy(),
            })

        self.classes = self.Y_predicted.unique().astype('str')
        self.classes.sort()

    def filter_classes(self, classes: List):
        '''Filter the dataframe categories with a list of classes.

        Parameters
        ----------
        classes : List
            List of allowed classes.

        '''
        self.classes = classes
        self.classes.sort()
        self.df = self.df[self.df['Y_predicted'].isin(self.classes)]

    @staticmethod
    def check_column_names(st: string):
        st = str(st)
        st = st.replace('/', '').replace(' ', '_').replace('-', '_')
        st = st.replace('?', 'unknown').replace('(', '').replace(')', '')
        st = ''.join([x.title() for x in st.split('_')])
        return st

    @staticmethod
    def check_var_names(st: string):
        '''Check for potential pyeda error in a string and automatically fix.
        '''
        banned_keywords = ['and', 'or', 'xor', 'not', 'nor', 'nand']
        st_complete = []
        for x in st.split(' '):
            if x.lower().strip() in banned_keywords:
                continue
            if x[0] in string.digits:
                x = 'z' + x
            st_complete.append(x)
        st = ' '.join(st_complete).title().replace('_', '')
        return st

    def generate_data_predictions(self, percent_dataset: int):
        pass

    def count_rule_occurrence(self, rule):
        return 1
