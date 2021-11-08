from time import time
import os
import random
from typing import Dict
from pydot import graph_from_dot_data

import numpy as np

from sklearn import tree
from sklearn import metrics
from sklearn.tree._tree import TREE_UNDEFINED
from sklearn.model_selection import RandomizedSearchCV
import contrxt.surrogate.generic_surrogate


class SklearnSurrogate(contrxt.surrogate.generic_surrogate.GenericSurrogate):
    '''
    '''

    def __init__(self, X, predicted_labels, time_label, class_id,
                 feature_names, hyperparameters):
        '''
        '''
        super().__init__(X, predicted_labels, time_label,
                         class_id, feature_names)

        self.hyperparameters = hyperparameters
        self.bdd = None
        self.paths = {}
        self.fit_time = None
        self.surrogate_predictions = None
        self.fidelity = None
        self._model = tree.DecisionTreeClassifier(
            splitter='best',
            criterion=self.hyperparameters['criterion'],
            min_samples_leaf=self.hyperparameters['min_samples_leaf'],
            max_depth=self.hyperparameters['max_depth'],
            min_samples_split=self.hyperparameters['min_samples_split'],
            # min_samples_leaf=min_samples_leaf,
            max_features=None, random_state=42,
            max_leaf_nodes=None,
            class_weight='balanced'
        )

    def hyperparameters_selection(self, param_grid: Dict=None, cv: int=5):
        '''
        '''
        start_time = time()
        np.random.seed(42)
        random.seed(42)
        self.logger.info('Beginning hyperparameters selection...')
        default_grid = {
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [0.01, 0.02],  # restrict the minimum number of samples in a leaf
            'max_depth': [3, 5, 7],  # helps in reducing the depth of the tree
            'min_samples_split': [0.01, 0.02, 0.03],  # restrict the minimum % of samples before splitting
        }
        search_space = param_grid if param_grid is not None else default_grid
        # Cost function aiming to optimize(Total Cost) = measure of fit + measure of complexity
        # References for pruning:
        # 1. http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # 2. https://www.coursera.org/lecture/ml-classification/optional-pruning-decision-trees-to-avoid-overfitting-qvf6v
        # Using Randomize Search here to prune the trees to improve readability without
        # comprising on model's performance
        verbose_level = 4 if self.logger.level >= 20 else 0
        random_search_estimator = RandomizedSearchCV(estimator=self._model, cv=cv,
                                                     param_distributions=search_space,
                                                     scoring='f1', n_iter=10, n_jobs=-1,
                                                     random_state=42, verbose=verbose_level)
        # train a surrogate DT
        random_search_estimator.fit(self.X, self.predicted_labels)
        # access the best estimator
        self._model = random_search_estimator.best_estimator_

        self.hyperparameters['max_depth'] = self._model.max_depth
        self.hyperparameters['min_samples_split'] = self._model.min_samples_split

        self.logger.info(f'Time for fitting surrogate: {round(time() - start_time, 3)}')
        self.logger.info(f'Best model: {self._model}')

    def fit(self):
        '''Fit the model.
        '''
        start_time = time()

        np.random.seed(42)
        random.seed(42)
        self._model.fit(self.X, self.predicted_labels)
        self.score()

        self.fit_time = round(time() - start_time, 3)
        self.logger.info(f'Time for fitting surrogate: {self.fit_time}')

    def score(self):
        '''Compute fidelity score.
        '''
        self.surrogate_predictions = self._model.predict(self.X)

        self.fidelity = {
            'f1_binary': metrics.f1_score(self.predicted_labels,
                                        self.surrogate_predictions,
                                        average='binary'),
            'f1_macro': metrics.f1_score(self.predicted_labels,
                                        self.surrogate_predictions,
                                        average='macro'),
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
        self.logger.info(metrics.classification_report(self.predicted_labels,
                                            self.surrogate_predictions))

    def surrogate_to_bdd_string(self):
        '''Transform surrogate to BDD string using depth first search.
        '''
        self.logger.info('Transforming surrogate to BDD string...')
        stack = []
        self.bdd = []

        def _tree_recurse(node):
            if self._model.tree_.feature[node] == TREE_UNDEFINED:
                # Leaf node, base case
                value = np.argmax(self._model.tree_.value[node][0])
                if value == 1:
                    path = ' & '.join(stack[:])
                    self.bdd.append(path)
                    self.paths[path] = self._model.tree_.n_node_samples[node]
                return

            # Recursion case
            name = self.feature_names[self._model.tree_.feature[node]]
            # self.logger.info(self.feature_names)
            stack.append(f'~{name}')
            self.logger.debug(stack)

            _tree_recurse(self._model.tree_.children_left[node])

            stack.pop()
            self.logger.debug(stack)
            stack.append(name)
            self.logger.debug(stack)

            _tree_recurse(self._model.tree_.children_right[node])

            stack.pop()
            self.logger.debug(stack)

        _tree_recurse(0)
        self.bdd = ' | '.join(self.bdd)
        self.logger.info(f'BDD String for class {self.class_id}: {self.bdd}')

    def save_surrogate_img(self, save_path):
        '''Save decision tree surrogates to image.
        '''
        directory = f'{save_path}/surrogate_tree'
        if not os.path.exists(directory):
            os.makedirs(directory)
        fname = f'{directory}/{self.class_id}_{self.time_label}.png'
        graph_str = tree.export_graphviz(
            self._model,
            class_names=[f'NOT {self.class_id}', self.class_id],
            feature_names=self.feature_names,
            filled=True
        )
        (graph,) = graph_from_dot_data(graph_str)
        self.logger.info(f'Saving {fname} to disk')
        graph.write_png(fname)
