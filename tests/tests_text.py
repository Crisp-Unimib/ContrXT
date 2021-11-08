import unittest
import sys
import os
import logging
import shutil

import pandas as pd

from tests.test_utils import setup_synthetic_text
from contrxt.trace import Trace
from contrxt.explain import Explain


class TestContrXTText(unittest.TestCase):

    def test_1_trace_instantiation(self):
        '''
        '''
        dir_path = os.path.dirname(os.path.realpath(__file__))
        shutil.rmtree(f'{dir_path}/test_results')
        os.mkdir(f'{dir_path}/test_results')
        (X_t1, Y_t1, predicted_labels_t1,
         X_t2, Y_t2, predicted_labels_t2) = setup_synthetic_text()

        self.assertEqual(len(X_t1), len(Y_t1), 'Different X and Y sizes in t1')
        self.assertEqual(len(X_t2), len(Y_t2), 'Different X and Y sizes in t2')
        self.assertEqual(len(Y_t1), len(predicted_labels_t1),
                         'Different prediction and Y sizes in t1')
        self.assertEqual(len(Y_t2), len(predicted_labels_t2),
                         'Different prediction and Y sizes in t2')

        trace = Trace(X_t1, Y_t1, predicted_labels_t1,
                      X_t2, Y_t2, predicted_labels_t2,
                      data_type='text', log_level=logging.DEBUG,
                      hyperparameters_selection=False,
                      save_path=f'{dir_path}/test_results',
                      save_surrogates=False, save_csvs=True)

    def test_2_trace_run(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        (X_t1, Y_t1, predicted_labels_t1,
         X_t2, Y_t2, predicted_labels_t2) = setup_synthetic_text()

        trace = Trace(X_t1, Y_t1, predicted_labels_t1,
                      X_t2, Y_t2, predicted_labels_t2,
                      data_type='text', log_level=logging.DEBUG,
                      hyperparameters_selection=False,
                      save_path=f'{dir_path}/test_results',
                      save_surrogates=True, save_csvs=True)

        # percent_mc = trace.run_montecarlo(threshold=0.5)
        # self.assertEqual(percent_mc, 0.2, f'Percent mc is {percent_mc}')

        trace.run_trace(1)
        n_classes = len(pd.read_csv(f'{dir_path}/test_results/trace.csv',
                        sep=';')['class_id'].unique())
        self.assertEqual(n_classes, 2,
                         'Error: expected 2 classes in csv, found {n_classes}.')

    def test_3_explain(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        explain = Explain(save_path=f'{dir_path}/test_results',
                          save_bdds=True, save_csvs=True)
        explain.run_explain()


if __name__ == '__main__':
    unittest.main()
