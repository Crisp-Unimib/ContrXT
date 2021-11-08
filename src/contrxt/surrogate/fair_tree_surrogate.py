from time import time
import os
import random
import copy
from typing import Dict

import numpy as np
import pandas as pd

import contrxt.surrogate.generic_surrogate
from sklearn import metrics


class FairTreeSurrogate(contrxt.surrogate.generic_surrogate.GenericSurrogate):
    '''
    '''

    def __init__(self, X, predicted_labels, time_label, class_id,
                 feature_names, hyperparameters):
        '''
        '''
        super().__init__(X, predicted_labels, time_label,
                         class_id, feature_names)

        self.bdd = None
        self.paths = {}
        self.fit_time = None
        self.surrogate_predictions = None
        self.fidelity = None
        self.hyperparameters = hyperparameters

    class Question:
        """A Question is used to partition a dataset.

        This class just records a 'column number' (e.g., 0 for Color) and a
        'column value' (e.g., Green). The 'match' method is used to compare
        the feature value in an example to the feature value stored in the
        question. See the demo below.
        """

        def __init__(self, column, value, feature_names):
            self.column = column
            self.value = value
            self.feature_names = feature_names
            self.feature = self.feature_names[self.column]

        def is_numeric(self, value):
            """Test if a value is numeric."""
            return isinstance(value, int) or isinstance(value, float)

        def match(self, example):
            # Compare the feature value in an example to the
            # feature value in this question.
            val = example[self.column]
            if self.is_numeric(val):
                return val >= self.value
            else:
                return val == self.value

        def __repr__(self):
            # This is just a helper method to print
            # the question in a readable format.
            condition = "=="
            if self.is_numeric(self.value):
                condition = ">="
            return "Is %s %s %s?" % (
                self.feature, condition, str(self.value))

    class Leaf:
        """A Leaf node classifies data.

        This holds a dictionary of class (e.g., "Apple") -> number of times
        it appears in the rows from the training data that reach this leaf.
        """

        def __init__(self, rows, threshold):
            dtf = pd.DataFrame(rows)
            support = len(rows)
            percent_positive = sum(dtf[dtf.columns[-1]]) / support
            stats = {
                "predicted": 1 if percent_positive > threshold else 0,
                "probability": percent_positive,
                "support": support,
            }

            # if sensitive_class and sensitive_feature:
                # stats["DP"]:  ((sum(dtf[dtf[sensitive_feature] == sensitive_class][-1]) / dtf[dtf[sensitive_feature] == sensitive_class].shape[0]) -
                #                (sum(dtf[dtf[sensitive_feature] != sensitive_class][-1]) / dtf[dtf[sensitive_feature] != sensitive_class].shape[0]))
            self.predictions = stats

    class DecisionNode:
        """A Decision Node asks a question.

        This holds a reference to the question, and to the two child nodes.
        """

        def __init__(self,
                     question,
                     true_branch,
                     false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

    @staticmethod
    def _partition(rows, question):
        """Partitions a dataset.

        For each row in the dataset, check if it matches the question. If
        so, add it to 'true rows', otherwise, add it to 'false rows'.
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)

        false = copy.deepcopy(false_rows)
        true = copy.deepcopy(true_rows)
        return true, false

    @staticmethod
    def _gini(rows):
        """Calculate the Gini Impurity for a list of rows.

        There are a few different ways to do this, I thought this one was
        the most concise. See:
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
        """
        def class_counts(rows):
            """Counts the number of each type of example in a dataset."""
            counts = {}  # a dictionary of label -> count.
            for row in rows:
                # in our dataset format, the label is always the last column
                label = row[-1]
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1
            return counts

        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity

    @staticmethod
    def _class_prob(rows):
        """Counts the target probability in the datasets"""
        counts = []
        for row in rows:
            counts.append(row[-1])
        return sum(counts)/len(counts)

    def _info_gain(self, left, right, current_uncertainty):
        """Information Gain.

        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self._gini(left) - (1 - p) * self._gini(right)

    def _info_DP(self, left, right, treshold):
        """Demographic Parity

        The DP introduced by the split.
        """
        P_left = self._class_prob(left)
        P_right = self._class_prob(right)
        pred_left = 0
        if P_left > treshold:
            pred_left = 1
        pred_right = 0
        if P_right > treshold:
            pred_right = 1
        for l in left:
            l = l.append(pred_left)
        for r in right:
            r = r.append(pred_right)
        dtf = pd.DataFrame(right).append(pd.DataFrame(left)).copy()
        dp = ((sum(dtf[dtf[8] == " Female"][14]) / dtf[dtf[8] == " Female"].shape[0]) -
                               (sum(dtf[dtf[8] == " Male"][14]) / dtf[dtf[8] == " Male"].shape[0]))
        return dp

    def _find_best_split(self, rows):
        """Find the best question to ask by iterating over every feature / value
        and calculating the information gain."""
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = self._gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = {row[col] for row in rows}  # unique values in the column

            for val in values:  # for each value

                question = self.Question(col, val, self.feature_names)

                # try splitting the dataset
                true_rows, false_rows = self._partition(rows, question)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = self._info_gain(true_rows, false_rows, current_uncertainty)

                #DP of the split
                # DP = abs(_info_DP(true_rows, false_rows, treshold = 0.2))

                # You actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our
                # toy dataset.
                if (gain >= best_gain): # & (DP<0.2)
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def _build_tree(self, rows):
        """Builds the tree.

        Rules of recursion: 1) Believe that it works. 2) Start by checking
        for the base case (no further information gain). 3) Prepare for
        giant stack traces.
        """
        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = self._find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if (gain < 0.005) | (len(rows) < (self.X.shape[0] * self.hyperparameters['min_samples_split'])):
            return self.Leaf(rows, self.hyperparameters['predict_threshold'])

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = self._partition(rows, question)

        # Recursively build the true branch.
        true_branch = self._build_tree(true_rows)

        # Recursively build the false branch.
        false_branch = self._build_tree(false_rows)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # dependingo on the answer.
        return self.DecisionNode(question, true_branch, false_branch)

    def predict(self, row, node):
        """See the 'rules of recursion' above."""

        # Base case: we've reached a leaf
        if isinstance(node, self.Leaf):
            return node.predictions["predicted"]

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.predict(row, node.true_branch)
        return self.predict(row, node.false_branch)

    def hyperparameters_selection(self, param_grid: Dict=None, cv: int=5):
        '''
        '''
        pass

    def fit(self):
        '''Fit the model.
        '''
        start_time = time()

        np.random.seed(42)
        random.seed(42)
        self._model = self._build_tree(pd.concat([self.X, pd.DataFrame(self.predicted_labels)], axis=1).values.tolist())
        self.score()

        self.fit_time = round(time() - start_time, 3)
        self.logger.info(f'Time for fitting surrogate: {self.fit_time}')

    def score(self):
        '''Compute fidelity score.
        '''
        self.surrogate_predictions = [self.predict(x[1], self._model) for x in self.X.iterrows()]

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
            if isinstance(node, self.Leaf):
                # Leaf node, base case
                value = node.predictions['predicted']
                if value == 1:
                    path = ' & '.join(stack[:])
                    self.bdd.append(path)
                    self.paths[path] = node.predictions['support']
                return

            name = node.question.feature
            stack.append(f'~{name}')
            self.logger.debug(stack)

            _tree_recurse(node.false_branch)

            stack.pop()
            self.logger.debug(stack)
            stack.append(f'{name}')
            self.logger.debug(stack)

            _tree_recurse(node.true_branch)

            stack.pop()
            self.logger.debug(stack)

        _tree_recurse(self._model)
        self.bdd = ' | '.join(self.bdd)
        self.logger.info(f'BDD String for class {self.class_id}: {self.bdd}')

    def save_surrogate_img(self, save_path):
        '''Save decision tree surrogates to image.
        '''
        self.print_tree(self._model)

    def print_tree(self, node, spacing=""):
        # Base case: we've reached a leaf
        if isinstance(node, self.Leaf):
            print (spacing + "Predict", node.predictions)
            return

        # Print the question at this node
        print (spacing + str(node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")
