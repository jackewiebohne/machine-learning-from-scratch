import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import copy

# b = (0,0)
# lam = lambda b,i,n: (b:=(i,n) if n>b[-1] else 0)
# x = np.array([2,3,4])
# # any(lam(b, i, n) for i,n in enumerate(x))
# [b:=(i,n) for i,n in enumerate(x) if n>b[-1]]
# print(b)

class tree:
    '''
    decision tree based on CART algorithm (though entropy is possible as criterion here)
    see e.g.: https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
    or for a really clear implementation: https://github.com/zziz/cart/blob/master/cart.py

    like scikit learn (which also uses CART) it doesn't handle categorical data.

    inputs: 
        tree_type: str: 'classification' or 'regression'
        measure: str: 'entropy', 'gini', or 'mse' (but 'mse' will also be selected by default if tree_type is 'regression')
        postpruning: tuple or None: performs the pruning based on a float parameter threshold and a measure given as str:
                                'mse', 'entropy', or 'gini', or 'misclassification'
        max_depth: int: maximum depth of tree (0-indexed) at which it will terminate 
        min_count: int: minimum number of samples for split

    note: underscore methods might have to be called with .root, because some assume that the tree object (with its root) has already 
          been created
    '''
    def __init__(self, tree_type='classification', measure='entropy', max_depth=4, min_count=5):
        assert (tree_type == 'classification' and measure in ('entropy', 'gini')) or (tree_type == 'regression')
        self.prune_dispatcher = {'mse': self._get_mse, 'gini': self._get_gini,
                      'entropy': self._get_entropy, 
                      'misclassification': self._get_missclassification}
        measure = measure
        self.tree_type = tree_type
        self.max_depth = max_depth
        self.min_count = min_count
        self.measure = measure
        self.depth = 0
        self.left = None
        self.right = None
        self.root = None
        self.n = None # number of samples at node
        self.impurity = None # impurity or mse at current node
        self.feature = None # feature at which we split
        self.gain = None # gain at best split
        self.label = None # the y-label (in case of regression this is simply the mean of y)
        self.threshold = None # numeric threshold that determines split

    def _get_entropy(self, y):
        unique_labels = np.unique(y)
        probs = np.array([len(y[y == label])/len(y) for label in unique_labels])
        return - np.sum(probs * np.log2(probs))

    def _get_gini(self, y):
        unique_labels = np.unique(y)
        probs = np.array([len(y[y == label])/len(y) for label in unique_labels])
        return 1 - np.sum(probs**2)
    
    def _get_mse(self, y):
        '''
        regression trees use the mean of y at a particular split
        against the actual y to compute the mean squared error
        '''
        return np.mean((y - np.mean(y)) ** 2)

    def _get_missclassification(self, y):
        '''
        'misclassification' is how it's called in the literature,
        but it's a bit of a misnomer tbh. more appropriately this is
        the majority-label proportion
        '''
        return 1 - max([len(y[y == label])/len(y) for label in np.unique(y)])

    def _get_impurity(self, y):
        if self.measure == 'entropy':
            return self._get_entropy(y)
        elif self.measure == 'gini':
            return self._get_gini(y)
        else: return self._get_mse(y)

    def _best_feature_split(self, x, y):
        self.n = x.shape[0]

        # need more than 1 sample and ground truth for split
        if self.n <= 1 or len(np.unique(y)) == 1:
            self.label = y[0]
            return

        if self.measure in ('gini', 'entropy'):
            # determine label by majority vote
            majority = (0,0)
            [majority := (label, len(y[y==label])) for label in np.unique(y) if len(y[y==label]) > majority[-1]]
            self.label = majority[0]
        else:
            self.label = np.mean(y)
        

        # pre-pruning/early stopping
        if self.depth + 1 > self.max_depth or self.n < self.min_count:
            return

        self.impurity = self._get_impurity(y)
        best_gain = 0
        best_feature = 0
        best_threshold = 0

        # since we're only working with numerical (not categorical) values
        # we sort values in current feature (which np.unique does automatically), 
        # so that the next highest rows for this feature are next to each other
        # then average the feature's row value with the next highest row value. this will give the thresholds to test.
        for feature in range(x.shape[1]):
            features = np.unique(x[:, feature]) # sorted & unique features
            thresholds = (features[1:] + features[:-1]) / 2 # get avg of row value to next highest row value 
            # initialise "best" variable which will hold: (best_gain, best_threshold)
            best = (0, 0)
            # use (ugly) list comprehension for efficiency; though it turns out it is not in fact faster in significant ways, presumably because there's a lot of slicing of the same thing that could have been saved in a variable in a for loop
            [best := (
                self.impurity - 
                ((len(y[x[:, feature] <= threshold])/self.n) * self._get_impurity(y[x[:, feature] <= threshold]) 
                + (len(y[x[:, feature] > threshold])/self.n) * self._get_impurity(y[x[:, feature] > threshold])),
                threshold 
                ) for threshold in thresholds 
                if best[0] < self.impurity - # node impurity
                ((len(y[x[:, feature] <= threshold])/self.n) * self._get_impurity(y[x[:, feature] <= threshold]) # proportion_left * left_impurity
                + (len(y[x[:, feature] > threshold])/self.n) * self._get_impurity(y[x[:, feature] > threshold]) )] # proportion_right * right_impurity

            if best[0] > best_gain:
                best_gain = best[0]
                best_threshold = best[1]
                best_feature = feature
        
        if self.gain == 0:
            return
        
        self.gain = best_gain
        self.feature = best_feature
        self.threshold = best_threshold

        # recursively split on left
        self.left = tree(self.tree_type, self.measure, self.max_depth, self.min_count)
        self.left.depth = self.depth + 1
        self.left.prev_impurity = self.impurity
        self.left._best_feature_split(x[x[:,self.feature] <= self.threshold], y[x[:,self.feature] <= self.threshold])
        # recursively split on right
        self.right = tree(self.tree_type, self.measure, self.max_depth, self.min_count)
        self.right.depth = self.depth + 1
        self.right.prev_impurity = self.impurity
        self.right._best_feature_split(x[x[:,self.feature] > self.threshold], y[x[:,self.feature] > self.threshold])

    def fit(self, x, y):
        self.root = tree(self.tree_type, self.measure, self.max_depth, self.min_count)
        self.root._best_feature_split(x, y)

    def _get_subtrees(self):
        pruned_trees = []
        def rec_prune(treepart, depth):
            if treepart.left is None and treepart.right is None:
                return  
            rec_prune(treepart.left, depth)
            rec_prune(treepart.right, depth)
            prune = False
            if treepart.depth >= depth:   
                prune = True
            if prune:
                treepart.right = None
                treepart.left = None
                treepart.feature = None

        pruned_tree = copy.deepcopy(self)
        depth = pruned_tree._get_max_depth()
        while depth > 0:
            pruned_tree = copy.deepcopy(pruned_tree)
            rec_prune(pruned_tree, depth)
            pruned_trees.append(pruned_tree)
            depth -= 1
        return pruned_trees

    def get_trees_with_alphas(self, prune_measure, x, y):
        '''
        for minimal cost complexity postpruning
        prune_measure: str: accepts 'mse', 'entropy', or 'gini', or 'misclassification' (this is 
                    the default in the literature and strongly recommended; the other measures
                    are made available for experimentation)
        x: np.array: training (or cross validation) data
        y: np.array: ground truth

        returns: a list of effectives alphas and their correspondingly pruned trees;
                 the alphas are to be understood such that:
                 the alpha alpha_best for which pruned_trees[i] is best is alphas[i] <= alpha_best < alphas[i+1]
        '''
        alphas = [0] # later on will change to most recent effective alpha, i.e. g_t[-1]
        pruned_trees = self._get_subtrees()
        return_trees = copy.deepcopy(pruned_trees)
        node_errors = self._get_node_errors(prune_measure, x, y)
        assert len(pruned_trees) == len(node_errors)
        while pruned_trees:
            g_t = []
            for i, tree in enumerate(pruned_trees):
                # print(tree)
                R_t = node_errors[i]
                total_error = pruned_trees[0]._get_leaf_errors(prune_measure, x, y) # most complete tree is always at beginning of list
                num_leaves = tree._get_num_leaves()
                eff_a = (R_t - total_error) / (num_leaves - 1)
                g_t.append(eff_a)
            # now find the smallest g_t and append to alphas
            # if there's a tie, choose the one that's closest to end of g_t
            # so we'll get the indices of the two smallest elements, and in case of tie
            # choose the smaller index so that we prune less (i.e. slice closer to beginning
            # of the tree list where we find the increasingly complete trees)
            g_t.reverse()
            check_min_ix = np.argsort(g_t)[:2]
            if len(check_min_ix) > 1 and np.allclose(g_t[check_min_ix[0]], g_t[check_min_ix[1]]):
                min_ix = min(check_min_ix)
            else:
                min_ix = check_min_ix[0]
            alphas.append(g_t[min_ix])
            if len(pruned_trees) > 1: pruned_trees = pruned_trees[min_ix+1:]
            else: return alphas, return_trees
        return alphas, return_trees

    def _get_leaf_errors(self, prune_measure, x, y):
        prune_measure = self.prune_dispatcher[prune_measure]
        def _recursive_error(treepart, prune_measure, error_list, x, y):
            if treepart.right is None and treepart.left is None: 
                error_list += [(prune_measure(y), len(y))]
            if treepart.left is not None:
                _recursive_error(treepart.left, prune_measure, error_list,
                                 x[x[:,treepart.feature] <= treepart.threshold],
                                 y[x[:,treepart.feature] <= treepart.threshold])
            if treepart.right is not None:
                _recursive_error(treepart.right, prune_measure,error_list,
                                 x[x[:,treepart.feature] > treepart.threshold],
                                 y[x[:,treepart.feature] > treepart.threshold])
        error_list = []
        _recursive_error(self, prune_measure, error_list, x, y)
        return np.sum([np.prod(ele) for ele in error_list]) / sum(ele[1] for ele in error_list)

    def _get_node_errors(self, prune_measure, x, y):
        '''
        for postpruning
        inputs:
            measure: a function to use (from the prune_dispatcher)
            y: labels
        '''
        prune_measure = self.prune_dispatcher[prune_measure]
        def _recursive_error(treepart, prune_measure, error_list, size_list, x, y):
            if treepart.feature is not None:
                error_list += [prune_measure(y)]
                size_list += [len(y)]
            if treepart.left is not None:
                _recursive_error(treepart.left, prune_measure, error_list, size_list,
                                 x[x[:,treepart.feature] <= treepart.threshold],
                                 y[x[:,treepart.feature] <= treepart.threshold])
            if treepart.right is not None:
                _recursive_error(treepart.right, prune_measure, error_list, size_list,
                                 x[x[:,treepart.feature] > treepart.threshold], 
                                 y[x[:,treepart.feature] > treepart.threshold])
            return size_list, error_list
        size_list, error_list = _recursive_error(self, prune_measure, error_list=[], size_list=[], x=x, y=y)
        return [e * s/self.n for s,e in zip(size_list, error_list)]

    def predict(self, x):
        def _predict_value(treepart, x):
            if treepart.left is not None and treepart.right is not None:
                if x[treepart.feature] <= treepart.threshold:
                    return _predict_value(treepart.left, x)
                else:
                    return _predict_value(treepart.right, x)
            else: 
                return treepart.label
        return np.array([_predict_value(self.root, sample) for sample in x])

    def _get_num_internal_nodes(self):
        count = 0
        if self.left is not None or self.right is not None:
            count += 1
        if self.left is not None:
            count += self.left._get_num_internal_nodes()
        if self.right is not None:
            count += self.right._get_num_internal_nodes()
        return count
        
    def _get_num_leaves(self):
        # returns: number of leaves of tree or subtree
        count = 0
        if self.left is None and self.right is None:
            # if self is definition of leaf increase count
            count += 1
        if self.left is not None:
            count += self.left._get_num_leaves()
        if self.right is not None:
            count += self.right._get_num_leaves()
        return count

    def _get_max_depth(self):
        # returns: maximum depth of tree or subtree
        depth = 0
        if self.left is None and self.right is None:
            # if self is definition of leaf increase count
            depth = self.depth
        if self.left is not None:
            t_depth = self.left._get_max_depth()
            depth = max(depth, t_depth)
        if self.right is not None:
            t_depth = self.right._get_max_depth()        
            depth = max(depth, t_depth)
        return depth

    def __str__(self):
        string = []
        def print_subroutine(treepart, string, depth):
            base = '----' * depth
            if treepart.left is None and treepart.right is None:
                string += [base + f' then: label: {treepart.label}, n samples: {treepart.n}']
                # print('returning')
                # print('\n'.join(string))
                return 
            if treepart.feature is not None:
                string += [base + f'if x[{treepart.feature}] <= {treepart.threshold}:']
                print_subroutine(treepart.left, string, depth+1)
                string += [base + f'elif x[{treepart.feature}] > {treepart.threshold}:']
                print_subroutine(treepart.right, string, depth+1)
        print_subroutine(self.root, string, 0)
        return '\n'.join(string)

########### Cost-Complexity Pruning example ###########
# example from: http://mlwiki.org/index.php/Cost-Complexity_Pruning
t = tree()
t.root = t
t.root.feature = 1 # 'Y'
t.root.threshold = 1.5
t.root.n = 16
t.root.depth = 0
# leaf
t.root.left = tree() 
t.root.left.label = 1
t.root.left.deth = 1

t.root.right = tree()
t.root.right.feature = 0 # 'X'
t.root.right.threshold = 2.5
t.root.right.depth = 1
# leaf
t.root.right.left = tree() 
t.root.right.left.label = 0
t.root.right.left.depth = 2

t.root.right.right = tree()
t.root.right.right.feature = 1 # 'Y'
t.root.right.right.threshold = 2.5
t.root.right.right.depth = 2
# leaves
t.root.right.right.left = tree()
t.root.right.right.left.label = 0
t.root.right.right.left.depth = 3
t.root.right.right.right = tree()
t.root.right.right.right.label = 1
t.root.right.right.right.depth = 3

x = np.array(
    # the following are: y = 0; i.e. the o symbols in the example
    [[1,2],
    [1,3],
    [1,4],
    [2,2],
    [2,3],
    [2,4],
    [3,2],
    [4,2],
    # y = 1; i.e. the black squares
    [1,1],
    [2,1],
    [3,1],
    [4,1],
    [3,3],
    [3,4],
    [4,3],
    [4,4]]
    )
y = np.hstack((np.zeros(8), np.ones(8)))
print(t.get_trees_with_alphas('misclassification', x, y))
# => effective alphas are: [0, 0.125, 0.125, 0.25]

########### Comparing Classification and Regression with Scikit-learn ###########
# example taken and amended from: https://github.com/zziz/cart/blob/master/cart.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree as sktree

def classification_example():
    print('\n\nClassification Tree')
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    cls = tree(tree_type='classification', measure='entropy', max_depth=3, min_count=2)
    cls.fit(X_train, y_train)
    print(cls)

    pred = cls.predict(X_test)
    print("This Classification Tree Prediction Accuracy:    {}".format(sum(pred == y_test) / len(pred)))
    # => 0.9736842105263158
    
    clf = sktree.DecisionTreeClassifier(criterion = 'entropy')
    clf = clf.fit(X_train, y_train)
    sk_pred = clf.predict(X_test)

    print("Sklearn Library Tree Prediction Accuracy:        {}".format(sum(sk_pred == y_test) / len(pred)))
    # => 0.9736842105263158
    # both trees thus have identical accuracy
    
def regression_example():
    print('\n\nRegression Tree')
    randy = np.random.RandomState(1)
    X = np.sort(randy.randn(1000, 1) * 4, axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - randy.rand(200))

    # Fit regression model
    reg = tree(tree_type='regression', measure='mse', max_depth=3, min_count=5)
    reg.fit(X, y)
    print(reg.root.max_depth)
    print(reg)

    pred = reg.predict(np.sort(4 * randy.rand(1, 1), axis = 0))
    print('This Regression Tree Prediction:            {}'.format(pred))
    # => [0.75196085]
    
    sk_reg = sktree.DecisionTreeRegressor(max_depth = 3)
    sk_reg.fit(X, y)
    sk_pred = sk_reg.predict(np.sort(4 * randy.rand(1, 1), axis = 0))
    print('Sklearn Library Regression Tree Prediction: {}'.format(sk_pred))
    # => [0.75196085]
    # both trees have identical predictions in this case

classification_example()
regression_example()
