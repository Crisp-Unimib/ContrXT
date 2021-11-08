import logging
from contrxt.util.logger import build_logger

class GenericSurrogate:
    '''
    Base object
    '''

    def __init__(self, X, predicted_labels, time_label,
                 class_id, feature_names):
        '''
        '''
        self.logger = build_logger(logging.INFO, __name__, 'logs/surrogate.log')

        self.X = X
        self.predicted_labels = predicted_labels
        self.time_label = time_label
        self.class_id = class_id
        self.feature_names = feature_names

    def hyperparameters_selection(self):
        pass

    def fit(self):
        pass

    def surrogate_to_bdd_string(self):
        pass

    def save_surrogate_img(self):
        pass
