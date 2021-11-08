import logging

from contrxt.trace import Trace
from contrxt.explain import Explain

from contrxt.data.text_data_manager import TextDataManager
from contrxt.data.tabular_data_manager import TabularDataManager

class ContrXT():
    '''
    '''

    def __init__(self,
                 X_t1, Y_t1, predicted_labels_t1,
                 X_t2, Y_t2, predicted_labels_t2,
                 data_type='text',
                 hyperparameters_selection=True,
                 log_level=logging.INFO,
                 save_path='results',
                 surrogate_type='sklearn',
                 save_surrogates=False,
                 save_csvs=True,
                 save_bdds=True):

        self.log_level = log_level
        self.hyperparameters_selection = hyperparameters_selection
        self.save_path = save_path
        self.surrogate_type = surrogate_type
        self.save_surrogates = save_surrogates
        self.save_csvs = save_csvs
        self.save_bdds = save_bdds

        if data_type == 'text':
            self.data_manager = {
                'time_1': TextDataManager(X_t1, Y_t1, predicted_labels_t1, 'time_1'),
                'time_2': TextDataManager(X_t2, Y_t2, predicted_labels_t2, 'time_2'),
            }
        else:
            self.data_manager = {
                'time_1': TabularDataManager(X_t1, Y_t1, predicted_labels_t1, 'time_1'),
                'time_2': TabularDataManager(X_t2, Y_t2, predicted_labels_t2, 'time_2'),
            }

        self.trace = None
        self.explain = None

    def run_trace(self, percent_dataset: float=1):
        '''
        '''
        self.trace = Trace(
            self.data_manager,
            log_level=self.log_level,
            hyperparameters_selection=self.hyperparameters_selection,
            save_path=self.save_path,
            surrogate_type=self.surrogate_type,
            save_surrogates=self.save_surrogates,
            save_csvs=self.save_csvs,
        )
        self.trace.run_trace(percent_dataset)

    def run_explain(self):
        '''
        '''
        self.explain = Explain(
            self.data_manager,
            save_path=self.save_path,
            log_level=self.log_level,
            save_bdds=self.save_bdds,
            save_csvs=self.save_csvs,
        )
        self.explain.run_explain()
