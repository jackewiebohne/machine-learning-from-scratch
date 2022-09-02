import numpy as np
from itertools import chain
from cart_tree import tree

class random_forest:
    '''
    based on Breiman's random forest (which uses CART decision trees also developed by Breiman)
    see: Breiman L (2001). "Random Forests". Machine Learning. 45 (1): 5–32.
    like scikit learn (which also uses CART) it doesn't handle categorical data.

    inputs: 
        tree_type: str: 'classification' or 'regression'
        n_estimators: int: number of trees
        measure: str: 'entropy', 'gini', or 'mse' (but 'mse' will also be selected by default if tree_type is 'regression')
        postpruning: tuple or None: performs the pruning based on a float parameter threshold and a measure given as str:
                                'mse', 'entropy', or 'gini', or 'misclassification'
        max_depth: int: maximum depth of tree (0-indexed) at which it will terminate 
        min_count: int: minimum number of samples for split
        random_features: int or None: 
                        recommended for classification: np.sqrt(x.shape[1])
                        recommended for regression: x.shape[1]//3
                        determines how big a subsample of the total available features should be taken at each
                        decision tree node. this is done because if one or a few features are very strong predictors
                        for the response variable (target output), these features will be selected 
                        in many of the trees, causing them to become correlated.
                        note that if random_features is equal to the total number of features it will still sample with replacement from
                        all features at each node, whereas it will not if None. 
        subsample: float or None: if a subsample (with replacement) of the training data should be used.
                        if None it uses the entire dataset, if any n such that 0 < n <= 1, it will use n % of the dataset.
                        note that if it is 1 it will still sample with replacement and thus the sampled dataset is unlikely to be
                        identical to the full dataset
        oob: bool: True or False. if those subsamples that aren't used for training (= out-of-bag) should be saved
                        for later approximations of the generalization error (see the method 'estimate_generalization_error')
        
    note: there are further options to use and modify the trees (e.g. minimal complexity pruning) using each tree's builtin 
          underscore methods. note, however, that they might have to be called with .root, because some of the methods
          assume that the tree object (with its root) has already been created
    '''
    def __init__(self, tree_type, n_estimators=200, measure='entropy', max_depth=4, min_count=5, random_features=None, subsample=None, oob=None):
        assert (tree_type == 'classification' and measure in ('entropy', 'gini')) or (tree_type == 'regression')
        self.tree_type = tree_type
        self.n_estimators = n_estimators
        self.measure = measure
        self.max_depth = max_depth
        self.min_count = min_count
        self.random_features = random_features
        self.subsample = subsample
        self.oob = oob
        # initialise the trees
        self.trees = [tree(self.tree_type, self.measure, self.max_depth, self.min_count, self.random_features) 
                      for _ in range(n_estimators)]
        # if we want to estimate generalization error we initialise an oob list for a range of oob indices per tree
        if oob is not None and subsample is not None: self.oob = []

    def fit(self, x, y):
        # some unsolicited recommendations ;-)
        if self.tree_type == 'classification' and (self.random_features is None or not np.allclose(np.sqrt(x.shape[1]), self.random_features)):
            print(f'please consider {np.sqrt(x.shape[1])} as input for self.random_features')
        elif self.tree_type == 'regression' and (self.random_features is None or np.allclose(x.shape[1]//3, self.random_features)):
            print(f'please consider {x.shape[1]//3} as input for self.random_features')
        if not self.subsample:
            print('please consider subsampling. it is recommended that self.subsample is 2/3 of the full data')

        for i in range(self.n_estimators):
            # subsample, otherwise use whole dataset
            if self.subsample is not None:
                sub_idx = np.random.choice(len(x), size=int(len(x) * self.subsample), replace=True)
                x_subsample = x[sub_idx]
                y_subsample = y[sub_idx]
                # if we want to estimate generalization error, we save the indices not in x_idx
                # for each tree so we can predict oob error later for each tree and aggregate
                # to get the generalization error
                if self.oob is not None:
                    # we use nonzero here because it will return the indices where the boolean array is true
                    # so we need to store a much smaller array rather than a boolean array the length of x
                    oob_ix = np.isin(np.arange(len(x)), sub_idx, invert=True).nonzero()
                    self.oob.append(oob_ix[0])#.tolist()) # because nonzero() returns a tuple, and since we'll be using the only as lists later
            else: 
                x_subsample = x
                y_subsample = y
            self.trees[i].fit(x_subsample, y_subsample)

    def predict(self, x):
        # preds = np.vstack([[tree.predict(sample) for tree in self.trees] for sample in x])
        preds = np.stack([tree.predict(x) for tree in self.trees], axis=1)
        if self.tree_type == 'classification':
            return np.array([np.bincount(row).argmax() for row in preds])
        else: return np.mean(preds, axis=1)

    def estimate_generalization_error(self, x, y):
        '''
        function to approximate the accuracy of the forest for out of sample data
        '''
        if not self.oob:
            print('random forest was not fit with oob == True and an appropriate value for subsample')
        # the oob indices might differ in length (due to sampling with replacement), 
        # => thus we have to work with lists, not arrays!
        # get the predictions of x
        len_x = len(x)
        x = [x[indices] for indices in self.oob]
        oob_preds = [self.trees[i].predict(x[i]).tolist() for i in range(len(self.trees))] # we can do this because the trees and oob lists are of same length
        ordered_oob_preds = [[] for _ in range(len_x)]

        # print(len())
        # for row_ix, row in enumerate(self.oob):
        #     for col_ix, col in enumerate(row):
        #         # print('row, col', row_ix, col_ix)
        #         print(col)
        #         appendit = oob_preds[row_ix][col_ix]
        #         ordered_oob_preds[col].append(appendit)

        any(ordered_oob_preds[col].append(oob_preds[row_ix][col_ix]) for row_ix, row in enumerate(self.oob) for col_ix, col in enumerate(row))
        ordered_oob_preds = [ele for ele in ordered_oob_preds if ele]
        # now get the right ys for the ground truth comparison
        unique_idx = list(set(chain(*self.oob)))
        y = y[unique_idx]
        if self.tree_type == 'classification':
            majority_vote = np.array([np.bincount(row).argmax() for row in ordered_oob_preds])
            print(f'approximate general classification accuracy: {sum(y == majority_vote) / len(y)}')
        else: 
            prediction_means = [np.mean(lst) for lst in ordered_oob_preds]
            mse = np.mean(np.sqrt((y-prediction_means)**2))
            y_range = np.max(y)-np.min(y)
            y_mean = np.mean(y)
            print(f'approximate general regression MSE: {mse}\n \
                    with a range of y values: {y_range}, and a mean of y values {y_mean} \n \
                    which gives proportion of y range to mse {mse/y_range} and proportion of mse to y mean {abs(mse/y_mean)}')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def classification_example():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    # print(X_test.shape)

    forest = random_forest(tree_type='classification', n_estimators=50, measure='entropy', max_depth=4, min_count=5, random_features=2, subsample=2/3, oob=True)
    forest.fit(X_train, y_train)

    pred = forest.predict(X_test)
    print("Classification Forest Prediction Accuracy: {}".format(sum(pred == y_test) / len(pred)))
    # Classification Forest Prediction Accuracy:    1.0
    forest.estimate_generalization_error(X_train, y_train)
    # approximate general classification accuracy: 0.9375

def regression_example():
    randy = np.random.RandomState(1)
    X = np.sort(randy.randn(1000, 1) * 4, axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - randy.rand(200))

    forest = random_forest(tree_type='regression', n_estimators=50, measure='mse', max_depth=3, min_count=5, random_features=2, subsample=2/3, oob=True)
    forest.fit(X, y)

    x_test = np.sort(4 * randy.randn(100, 1), axis = 0)
    y_test = np.sin(x_test).ravel() 
    y_test[::5] += 3 * (0.5 - randy.rand(20))
    
    pred = forest.predict(x_test)
    print('Regression Tree Prediction MSE: {}'.format( np.mean(np.sqrt((y_test-pred)**2))))
    # This Regression Tree Prediction mse: 0.3284577649008208
    forest.estimate_generalization_error(X, y)
    # approximate general regression MSE: 0.32890628351818996
    # with a range of y values: 4.748878783141422, and a mean of y values -0.022932134404935297 
    # which gives proportion of y range to mse 0.06925977657838125 and proportion of mse to y mean 14.342593572424073

classification_example()
# regression_example()