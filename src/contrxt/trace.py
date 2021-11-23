from typing import List, Tuple
import re
from time import time
from collections import defaultdict
import logging
import traceback
import os

import numpy as np
import pandas as pd

from contrxt.util.helpers import jaccard_similarity
from contrxt.util.logger import build_logger
from contrxt.surrogate.sklearn_surrogate import SklearnSurrogate


class Trace:
    '''Trace.

    Parameters
    ----------
    data_type : str
        Choose to use text or tabular data.
    hyperparameters_selection : bool
        Enables tuning of hyperparameters (the default is True).
    log_level :
        Choose logging level (INFO, DEBUG...).
    save_path : str
        Path for saving csv results (the default is 'results').
    save_surrogates : bool
        Toggle for saving tree images (the default is False).
    save_csvs : bool
        Toggle for saving csv files (the default is True).

    Attributes
    ----------

    '''

    def __init__(self,
                 data_manager,
                 hyperparameters_selection=True,
                 log_level=logging.INFO,
                 save_path='results',
                 surrogate_type='sklearn',
                 save_surrogates=False,
                 save_csvs=True):
        '''
        Initialize class Trace
        '''

        self.logger = build_logger(log_level, __name__, 'logs/trace.log')

        self.data_manager = data_manager
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_surrogates, self.save_csvs = save_surrogates, save_csvs
        self.surrogate_type = surrogate_type

        self.hyperparameters_selection = hyperparameters_selection

        self._assign_classes()
        self._initialize_hyperparameters()
        self._initialize_result_dicts()

    def _assign_classes(self):
        '''Get list of common classes.
        '''
        classes_t1 = set(self.data_manager['time_1'].classes)
        classes_t2 = set(self.data_manager['time_2'].classes)
        self.classes = np.array(list(classes_t1.intersection(classes_t2)))
        self.classes.sort()

        self.logger.info(f'List of common classes: {self.classes}')

        self.data_manager['time_1'].filter_classes(self.classes)
        self.data_manager['time_2'].filter_classes(self.classes)

    def _initialize_result_dicts(self):
        '''Initialize result dictionaries.
        '''
        self.bdds = {'time_1': {}, 'time_2': {}}
        self.paths = {'time_1': {}, 'time_2': {}}
        self.times = {'time_1': {}, 'time_2': {}}
        self.fidelities = {'time_1': {}, 'time_2': {}}

    def _initialize_hyperparameters(self):
        '''Initialize hyperparameters dictionary.
        '''
        if self.surrogate_type == 'fair':
            self.hyperparameters = {
                'time_1': defaultdict(lambda: {'max_depth': 5, 'min_samples_split': 0.02,
                                              'predict_threshold': 0.5, 'sensitive_class': None,
                                              'sensitive_feature': None}),
                'time_2' : defaultdict(lambda: {'max_depth': 5, 'min_samples_split': 0.02,
                                              'predict_threshold': 0.5, 'sensitive_class': None,
                                              'sensitive_feature': None}),
            }
        else:
            self.hyperparameters = {
                'time_1': defaultdict(lambda: {'max_depth': 5, 'min_samples_split': 0.02,
                                               'criterion': 'gini', 'min_samples_leaf': 0.01}),
                'time_2': defaultdict(lambda: {'max_depth': 5, 'min_samples_split': 0.02,
                                               'criterion': 'gini', 'min_samples_leaf': 0.01}),
            }

    def _generate_trees(self, time_label: str):
        '''Generate surrogate tree.

        Parameters
        ----------
        time_label : str
            Description of parameter `time_label`.

        '''
        for class_id in self.classes:
            try:
                self.logger.info(f'Starting explanation in {time_label} for class_id {class_id}')
                start_time = time()

                if self.surrogate_type == 'fair':
                    surrogate_explainer = FairTreeSurrogate(
                        self.data_manager[time_label].surrogate_train_data[class_id],
                        self.data_manager[time_label].Y_predicted_binarized[class_id],
                        time_label, class_id,
                        self.data_manager[time_label].feature_names,
                        self.hyperparameters[time_label][class_id],
                    )
                else:
                    surrogate_explainer = SklearnSurrogate(
                        self.data_manager[time_label].surrogate_train_data[class_id],
                        self.data_manager[time_label].Y_predicted_binarized[class_id],
                        time_label, class_id,
                        self.data_manager[time_label].feature_names,
                        self.hyperparameters[time_label][class_id],
                    )

                if self.hyperparameters_selection:
                    surrogate_explainer.hyperparameters_selection()
                    self.hyperparameters[time_label][class_id] = surrogate_explainer.hyperparameters
                surrogate_explainer.fit()
                surrogate_explainer.surrogate_to_bdd_string()

                self.bdds[time_label][class_id] = surrogate_explainer.bdd
                self.paths[time_label][class_id] = surrogate_explainer.paths
                self.fidelities[time_label][class_id] = surrogate_explainer.fidelity

                if self.save_surrogates:
                    surrogate_explainer.save_surrogate_img(self.save_path)

                self.times[time_label][class_id] = round(time() - start_time, 3)

            except Exception:
                self.logger.exception(traceback.print_exc())
                break

    def run_trace(self, percent_dataset: float=1):
        '''Run Trace.

        Parameters
        ----------
        percent_dataset : int
            Description of parameter `percent_dataset` (the default is 1).
        select_hyperparameters : type
            Description of parameter `select_hyperparameters` (the default is False).

        '''

        for time_label in ['time_1', 'time_2']:
            self.data_manager[time_label].generate_data_predictions(percent_dataset)
            self._generate_trees(time_label)

        if self.save_csvs:
            self._save_results(percent_dataset)
            self._save_surrogate_paths()

    @staticmethod
    def _str_to_list_features(original_string: str):
        '''Short summary.

        Parameters
        ----------
        original_string : str
            Description of parameter `original_string`.

        Returns
        -------
        out: List
            Description of returned object.

        '''
        if isinstance(original_string, list):
            return original_string
        characters_to_remove = '|&~'
        pattern = '[' + characters_to_remove + ']'
        return re.sub(' +', ' ', re.sub(pattern, '', original_string)).split(' ')

    def _compare_bdd_j(self, bdd1, bdd2):
        '''Short summary.

        Parameters
        ----------
        bdd1 : bdd
            Description of parameter `bdd1`.
        bdd2 : bdd
            Description of parameter `bdd2`.

        Returns
        -------
        out: Dict
            Description of returned object.

        '''
        j_dict = {'time_1': {}, 'time_2': {}}
        for time_, values in bdd1.items():
            for class_id, _ in values.items():
                bdd1[time_][class_id] = self._str_to_list_features(bdd1[time_][class_id])
                bdd2[time_][class_id] = self._str_to_list_features(bdd2[time_][class_id])
                j_dict[time_][class_id] = jaccard_similarity(bdd1[time_][class_id],
                                                             bdd2[time_][class_id])

        self.logger.debug(f'Jaccard similarity: {j_dict}')

        j_dict = {
            'time_1': {
                'mean': round(np.array(list(j_dict['time_1'].values())).mean(), 3),
                'std': round(np.array(list(j_dict['time_1'].values())).std(), 3),
            },
            'time_2': {
                'mean': round(np.array(list(j_dict['time_2'].values())).mean(), 3),
                'std': round(np.array(list(j_dict['time_2'].values())).std(), 3),
            }
        }
        return j_dict

    def run_montecarlo(self, threshold: int=0.9):
        '''Short summary.

        Parameters
        ----------
        threshold : type
            Description of parameter `threshold` (the default is 0.9).

        Returns
        -------
        type
            Description of returned object.

        '''
        mc_results = {}
        for i in range(10):
            percent_dataset = round(i / 10 + 0.1, 2)

            self.logger.info(f'----------\nStarting montecarlo with {percent_dataset*100}% of dataset.')

            previous_bdds = self.bdds.copy()
            self._initialize_result_dicts()

            self.run_trace(percent_dataset)
            self.hyperparameters_selection = False

            if previous_bdds['time_1']:
                j_dict = self._compare_bdd_j(previous_bdds, self.bdds)

                mean_j_time_1 = j_dict['time_1']['mean']
                std_j_time_1 = j_dict['time_1']['std']
                mean_j_time_2 = j_dict['time_2']['mean']
                std_j_time_2 = j_dict['time_2']['std']
                mc_results[percent_dataset] = {
                    'percentage_1': round((percent_dataset - 0.1) * 100),
                    'percentage_2': round(percent_dataset * 100),
                    'mean_j_time_1': mean_j_time_1,
                    'std_j_time_1': std_j_time_1,
                    'mean_j_time_2': mean_j_time_2,
                    'std_j_time_2': std_j_time_2,
                }

                self.logger.info((f'J for time_1: {mean_j_time_1}+-{std_j_time_1}, '
                             f'time_2: {mean_j_time_2}+-{std_j_time_2}.'))

                if (mean_j_time_1 > threshold) & (mean_j_time_2 > threshold):
                    self.logger.info('J values are good, im out')
                    break
        mc_results = pd.DataFrame.from_dict(mc_results, orient='index',
                                    columns=['percentage_1', 'percentage_2', 'mean_j_time_1',
                                             'std_j_time_1', 'mean_j_time_2', 'std_j_time_2'])
        mc_results.to_csv(f'{self.save_path}/montecarlo.csv', index=False, decimal='.', sep=';')
        return percent_dataset

    def _save_results(self, percent_dataset):
        '''Save results to csv.
        '''
        bdd_df = []
        for time_label in ['time_1', 'time_2']:
            for class_id in self.classes:
                class_id = str(class_id)
                try:
                    row = (
                        time_label, class_id,
                        self.bdds[time_label][class_id],
                        self.times[time_label][class_id],
                        percent_dataset,
                        *self.hyperparameters[time_label][class_id].values(),
                        *self.fidelities[time_label][class_id].values(),
                    )
                    bdd_df.append(row)

                except KeyError:
                    continue
        hyperparameters_cols = list(self.hyperparameters['time_1'][self.classes[0]].keys())
        fidelities_cols = list(self.fidelities['time_1'][self.classes[0]].keys())
        bdd_df = pd.DataFrame(bdd_df, columns=['time_label', 'class_id', 'bdd_string', 'runtime', 'percent_dataset'] +
                                              hyperparameters_cols + fidelities_cols)
        self.logger.info(f'Mean fidelity: {round(bdd_df[fidelities_cols[0]].mean(), 3)}')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        try:
            file_name = f'{self.save_path}/trace.csv'
            self.logger.info(f'Saving Trace results to {file_name}')
            bdd_df.to_csv(file_name, index=False, decimal='.', sep=';')
        except PermissionError:
            self.logger.error('Error: cannot save results, file in use!')

    def _save_surrogate_paths(self):
        '''Save surrogate paths to csv file.
        '''
        cols = ['time_label', 'class_id', 'bdd_string', 'n']
        paths_df = pd.DataFrame([], columns=cols)
        for time_label in ['time_1', 'time_2']:
            for class_id in self.classes:
                class_id = str(class_id)
                try:
                    df_paths_class = pd.DataFrame.from_dict(self.paths[time_label][class_id],
                                                            orient='index')
                    df_paths_class = df_paths_class.reset_index()
                    df_paths_class.columns = ['bdd_string', 'n']
                    df_paths_class['class_id'] = class_id
                    df_paths_class['time_label'] = time_label
                    df_paths_class = df_paths_class[cols]
                    paths_df = pd.concat([paths_df, df_paths_class])

                except KeyError:
                    continue

        try:
            file_name = f'{self.save_path}/surrogate_paths.csv'
            self.logger.info(f'Saving Surrrogate paths to {file_name}')
            paths_df.to_csv(file_name, index=False, decimal='.', sep=';')
        except PermissionError:
            self.logger.error('Error: cannot save surrogate paths, file in use!')
