from time import time
import os
from typing import Dict
#from pydot import graph_from_dot_data
import numpy as np
import pandas as pd
import contrxt.surrogate.generic_surrogate
from rulefit import RuleFit
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV



class RulefitSurrogate(contrxt.surrogate.generic_surrogate.GenericSurrogate):
    '''
    '''

    def __init__(self, X, predicted_labels, time_label, class_id,
                 feature_names, hyperparameters):
        '''
        '''
        X = pd.DataFrame.sparse.from_spmatrix(X).values
        super().__init__(X, predicted_labels, time_label,
                         class_id, feature_names)

        self.hyperparameters = {}
        self.bdd = None
        self.paths = None
        self.fit_time = None
        self.surrogate_predictions = None
        self.fidelity = None
        self._model = RuleFit()

    def hyperparameters_selection(self, param_grid: Dict = None, cv: int = 5):
        '''
        '''
        pass

    def fit(self):
        '''Fit the model.
        '''
        start_time = time()

        self._model.fit(self.X, self.predicted_labels, feature_names=self.feature_names)
        self.score()

        self.fit_time = round(time() - start_time, 3)
        self.logger.info(f'Time for fitting surrogate: {self.fit_time}')

    def score(self):
        '''Compute fidelity score.
        '''

        pred = self._model.predict(self.X)
        # converting the predictions to 0 and 1
        self.surrogate_predictions = np.array([1 if x >=0.5 else 0 for x in pred])

        self.fidelity = {
            'f1_binary': metrics.f1_score(self.predicted_labels,
                                          self.surrogate_predictions,
                                          average='binary'),
            'f1_weighted': metrics.f1_score(self.predicted_labels,
                                            self.surrogate_predictions,
                                            average='weighted'),
            'recall_weighted': metrics.recall_score(self.predicted_labels,
                                                    self.surrogate_predictions,
                                                    average='weighted'),
            'precision_weighted': metrics.precision_score(self.predicted_labels,
                                                          self.surrogate_predictions,
                                                          average='weighted'),
            'balanced_accuracy': metrics.balanced_accuracy_score(self.predicted_labels,
                                                                 self.surrogate_predictions),
        }
        self.fidelity = {k: round(v, 3) for k, v in self.fidelity.items()}

        self.logger.debug(self.predicted_labels[:100])
        self.logger.debug(self.surrogate_predictions[:100])
        self.logger.info(f'Fidelity of the surrogate: {self.fidelity}')
        self.logger.info(metrics.classification_report(self.predicted_labels, self.surrogate_predictions))

    def surrogate_to_bdd_string(self):
        '''Transform surrogate to BDD string using depth first search.
        '''
        def prune(rule_df):
            """
            prunes the rules to get only the top 5%
            """
            upper = np.percentile(rule_df['coef'], 95)
            res = rule_df[rule_df['coef'] >= upper]
            return res

        def formatter(text):
            """
            Transforms rulefit rules texts to the correct format
            e.g:
            charge_clinic <= 0.5 & fortunately <= 0.5 --> ~charge_clinic & ~fortunately
            """
            features = text.split('&')
            res = ''
            for feature in features:
                import re
                feature_spl = re.split('[> < <= >=]', feature)
                feature_spl = [x for x in feature_spl if x != ''][0]
                if '<' in feature:
                    res += f' ~{feature_spl} &'
                elif '>' in feature:
                    res += f' {feature_spl} &'
            return res.rstrip('&').strip()

        def create_path(rule_df):
            """
            concatenate rules with '|'
            """
            df = prune(rule_df)
            rules = df['rule']
            res = []
            for rule in rules:
                res.append(formatter(rule))
            return ' | '.join(res)

        self.logger.info('Transforming surrogate to BDD string...')

        self.bdd = []
        rules = self._model.get_rules()
        rules = rules[(rules.coef != 0) & (rules['type'] == 'rule')]
        df_rule = pd.DataFrame(rules)

        self.bdd = create_path(df_rule)
        self.logger.debug(f'BDD String for class {self.class_id}: {self.bdd}')

    def save_surrogate_img(self, save_path):
        '''Save decision tree surrogates to image.
        '''
        pass
