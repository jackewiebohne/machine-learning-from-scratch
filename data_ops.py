# data operations
import numpy as np

def shuffle_data(x, y, seed=None):
    '''
    shuffles data randomly (assuming it fits in memory)
    seed can be provided as int
    '''
    if seed:
        np.random.seed(seed)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    return x[idx], y[idx]

def k_splits(k, x, y, shuffle=False, seed=None):
    '''
    performs k-many splits for purposes of cross validation
    assumes that x and y fit in memory
    is a generator that yields a new 
    
    inputs:
        k : int: number of splits to perform on data
        x : np.array (2D): training data and features with shape(samples, features)
        y : np.array: ground truth
        shuffle : bool : if the data should be shuffled before k_splits
        seed : int : random seed

    outputs: 
        2 tuples of 2 arrays each:
            (x_train, y_train), (x_test, y_test)
    '''
    if shuffle:
        x, y = shuffle_data(x, y, seed)
    assert k > 1
    n = len(x)
    n_splits = int(np.ceil(n/k))
    assert n_splits >= 1
    for i in range(0, n, n_splits):
        begin, end = i, min(i+n_splits, n)
        test_ix = np.arange(n)[begin:end]
        train_ix = np.invert(np.isin(np.arange(n), test_ix))
        yield (x[train_ix], y[train_ix]), (x[test_ix], y[test_ix])



def to_categorical():
    pass

def normalize():
    pass

def train_test_split(x, y, test_size=0.1, shuffle=False, seed=None):
    '''       
    inputs:
        x : np.array (2D): training data and features with shape(samples, features)
        y : np.array: ground truth
        test_size : float: size of test set
        shuffle : bool : if the data should be shuffled 
        seed : int : random seed

    outputs: 
        2 tuples of 2 arrays each:
            (x_train, y_train), (x_test, y_test)
    '''
    n = len(x)
    ix = int(n * test_size)
    if shuffle:
        x, y = shuffle_data(x, y, seed)
    return (x[ix:], y[ix:]), (x[:ix], y[:ix])

# TODO: make shuffling a choice
def generate_batches(x, y, batch_size):
    '''
    generates batches of n-dimensional arrays that are shuffled
    note that the first dimension of the array MUST be the batches (not RBG channels etc.)
    '''
    assert batch_size <= len(x)
    if batch_size != len(x):
        choices = set(range(len(x))) 
        for i in range(0, len(x)-batch_size, batch_size):
            if len(choices) >= batch_size:
                shuffler = np.random.choice(list(choices), size=batch_size, replace=False)
                choices.difference_update(set(shuffler))
                x_batch = x[shuffler, ...]
                y_batch = y[shuffler]
                yield (x_batch, y_batch)
        yield (x[list(choices), ...], y[list(choices)]) # otherwise the generator will forget about the last batch
    else: yield (x, y)
