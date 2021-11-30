import unittest
import os
import shutil

import pandas as pd

from tests.test_utils import setup_synthetic_tabular
from contrxt.contrxt import ContrXT
from contrxt.data.tabular_data_manager import TabularDataManager

dir_path = os.path.dirname(os.path.realpath(__file__))
(X_t1, predicted_labels_t1,
 X_t2, predicted_labels_t2) = setup_synthetic_tabular()


class TestContrXTTabular(unittest.TestCase):

    def test_0_data_manager(self):
        '''
        '''
        if os.path.exists(f'{dir_path}/test_results'):
            shutil.rmtree(f'{dir_path}/test_results')
        os.mkdir(f'{dir_path}/test_results')

        self.assertEqual(len(X_t1), len(predicted_labels_t1),
                         'Different prediction and X sizes in t1')
        self.assertEqual(len(X_t2), len(predicted_labels_t2),
                         'Different prediction and X sizes in t2')

        TabularDataManager(X_t1, predicted_labels_t1, 'time_1')
        TabularDataManager(X_t2, predicted_labels_t2, 'time_2')

    def test_1_trace_instantiation(self):
        '''
        '''
        ContrXT(
            X_t1, predicted_labels_t1,
            X_t2, predicted_labels_t2,
            data_type='tabular', surrogate_type='sklearn',
            hyperparameters_selection=False,
            save_path=f'{dir_path}/test_results',
            save_surrogates=False, save_bdds=False
        )

    def test_2_run_trace(self):
        '''
        '''
        exp = ContrXT(
            X_t1, predicted_labels_t1,
            X_t2, predicted_labels_t2,
            data_type='tabular', surrogate_type='sklearn',
            hyperparameters_selection=False,
            save_path=f'{dir_path}/test_results',
            save_surrogates=False, save_bdds=False
        )

        percent_mc = exp.trace.run_montecarlo(threshold=0.5)
        self.assertEqual(percent_mc, 0.2, f'Percent mc is {percent_mc}')
        exp.run_trace(1)

        n_classes = len(pd.read_csv(f'{dir_path}/test_results/trace.csv',
                        sep=';')['class_id'].unique())
        self.assertEqual(n_classes, 2,
                         f'Error: expected 2 classes in csv, found {n_classes}.')

    def test_3_explain(self):
        '''
        '''
        exp = ContrXT(
            X_t1, predicted_labels_t1,
            X_t2, predicted_labels_t2,
            data_type='tabular', surrogate_type='sklearn',
            hyperparameters_selection=False,
            save_path=f'{dir_path}/test_results',
            save_surrogates=False, save_bdds=False
        )
        exp.run_explain()


if __name__ == '__main__':
    unittest.main()
