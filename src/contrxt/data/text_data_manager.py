import logging
import ast
import re
from contrxt.util.logger import build_logger
import contrxt.data.data_manager

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

class TextDataManager(contrxt.data.data_manager.DataManager):
    '''
    '''

    def __init__(self, X, Y, predicted_labels, time_label):
        '''
        '''
        X = pd.Series([self.check_var_names(x) for x in X])
        super().__init__(X, Y, predicted_labels, time_label)

    def generate_data_predictions(self, percent_dataset: int):
        '''Generated predictions for both datasets.

        Parameters
        ----------
        percent_dataset : int
            Describes the percentage of the dataset to use.
        '''
        self.logger.info(f'Sampling dataset with percent: {percent_dataset} and saving labels...')
        self.logger.info(f'Total dataset n is {self.df.shape[0]}.')

        data = self.df.sample(frac=percent_dataset, replace=False, random_state=42)
        self.logger.info(f'N. Samples {self.time_label}: {data.shape[0]}')

        onehot_vectorizer = CountVectorizer(binary=True, lowercase=False)

        onehot_vectorizer.fit(data['X'])
        self.feature_names = onehot_vectorizer.get_feature_names()

        for i, class_id in enumerate(self.classes):

            data_class = data.copy()
            data_class['Y'] = data_class['Y'].astype('str')
            class_id = str(class_id)
            # Balancing
            n_positive_class = sum(data_class['Y'] == class_id)
            try:
                data_class = pd.concat([
                    data_class[data_class['Y'] == class_id],
                    data_class[data_class['Y'] != class_id].sample(n=n_positive_class, replace=False, random_state=42),
                ])
            except ValueError:
                pass
            # data_class.to_csv(f'results/{self.time_label}_{class_id}.csv', index=False)

            self.surrogate_train_data[class_id] = onehot_vectorizer.transform(data_class['X'])

            try:
                true_labels = preprocessing.label_binarize(data_class['Y'], classes=self.classes)
                col_position = np.where(self.classes == class_id)[0][0]
                self.Y[class_id] = true_labels[:, col_position]
            except:
                self.Y[class_id] = data_class['Y']

            if data_class['predicted_labels'].dtype == np.dtype(np.int) or data_class['predicted_labels'].dtype == np.dtype(np.int64):
                self.predicted_labels_binarized[class_id] = np.array([1 if int(x) == i else 0 for x in data_class['predicted_labels']])
            else:
                self.predicted_labels_binarized[class_id] = np.array([1 if x == class_id else 0 for x in data_class['predicted_labels']])

        self.logger.info(f'Finished predicting {self.time_label}')

    def count_rule_occurrence(self, rule):
        '''Count the number of occurrences in the corpus for a specific rule.
        '''
        count = 0
        rule_dict = ast.literal_eval(re.sub(r'(\w+):', r'"\1":', rule))
        contains = [elem[0] for elem in rule_dict.items() if elem[1] == 1]
        avoids = [elem[0] for elem in rule_dict.items() if elem[1] == 0]
        for sentence in self.X:
            if (all(x in sentence.split() for x in contains)
                and all(x not in sentence.split() for x in avoids)):
                count += 1
        return count
