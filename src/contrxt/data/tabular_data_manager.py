import logging
from contrxt.util.logger import build_logger
import contrxt.data.data_manager

import numpy as np
import pandas as pd

from sklearn import preprocessing

class TabularDataManager(contrxt.data.data_manager.DataManager):
    '''
    '''

    def __init__(self, X, Y, predicted_labels, time_label):
        '''
        '''
        X.columns = [self.check_var_names(x) for x in X.columns]
        super().__init__(X, Y, predicted_labels, time_label)

    def generate_data_predictions(self, percent_dataset: int):
        '''Generated predictions for both datasets.

        Parameters
        ----------
        percent_dataset : int
            Describes the percentage of the dataset to use.

        '''
        self.logger.info(f'Sampling dataset with percent: {percent_dataset} and saving labels...')

        data = self.df.sample(frac=percent_dataset, replace=False, random_state=42)
        self.logger.info(f'N. Samples {self.time_label}: {data.shape[0]}')

        self.feature_names = list(self.X.columns)

        max_n = data['Y'].value_counts().min()

        for i, class_id in enumerate(self.classes):

            data_class = data.copy()
            data_class['Y'] = data_class['Y'].astype('str')
            class_id = str(class_id)
            # Balancing
            data_class = pd.concat([
                data_class[data_class['Y'] == class_id].sample(n=max_n, replace=False, random_state=42),
                data_class[data_class['Y'] != class_id].sample(n=max_n, replace=False, random_state=42),
            ]).reset_index(drop=True)
            # data_class.to_csv(f'results/{self.time_label}_{class_id}.csv', index=False)

            assert sum(data_class['Y'] == class_id) == sum(data_class['Y'] != class_id)

            self.surrogate_train_data[class_id] = data_class[self.feature_names].copy()

            try:
                true_labels = preprocessing.label_binarize(data_class['Y'], classes=self.classes)
                col_position = np.where(self.classes == class_id)[0][0]
                self.Y[class_id] = true_labels[:, col_position].copy()
            except:
                self.Y[class_id] = data_class['Y'].copy()

            if data_class['predicted_labels'].dtype == np.dtype(np.int) or data_class['predicted_labels'].dtype == np.dtype(np.int64):
                self.predicted_labels_binarized[class_id] = np.array([1 if int(x) == i else 0 for x in data_class['predicted_labels']])
            else:
                self.predicted_labels_binarized[class_id] = np.array([1 if x == class_id else 0 for x in data_class['predicted_labels']])

        self.logger.info(f'Finished predicting {self.time_label}')

    def count_rule_occurrence(self, rule):
        return 1
