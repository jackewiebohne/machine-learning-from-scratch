# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

class knn:
    def __init__(self, k, x_train, y_train, method='exact', measure='euclid', verbose=0, *args):
        '''
        inputs:
            k : int: number of neirest neighbors to consider
            x_train : array:  shape(n_samples, features/dims)       
            y_train : 1D array: labels
            method : str: if 'exact' then calculates exact k neighbors
                          if 'approximate' it generates hyperplanes, and uses
                          the sign of the dot product of hyperplanes with x_train
                          to create a hash dict of indices of x_train (the setup time
                          will be magnitudes bigger than exact (which is already truly fast, because
                          vectorized) and it'll only pay off with truly large sample size).
                          at prediction time it'll then get the sign of the dot product of the planes 
                          with x_test as hash value to find the x_train that occupy a similar
                          subspace (if no x_train with similar subspace are found it reverts to
                          the exact method). It should be noted that hashing is only efficient 
                          if the class labels are distributed in similar ways; e.g. if 
                          x1 = np.random.randn(200,2) and x2 = np.random.randn(200,2) * 0.2 - 5 
                          then the x2 will be hashed effectively under the same hash value and
                          only x1 will be split up (since it is distributed across a greater
                          area of space) => hashing is not very efficient here!
                             for increased accuracy, we could also create several sets of hyperplanes, 
                             hash x_train accordingly, and then match x_test against those and calculate the
                             closest. but this would be somewhat pointless for the purposes here, not least
                             given that the exact method is very fast
            measure : str: if 'euclid' (= default) uses traditional euclidian distance
                           else uses cosine distance, euclid is highly recommended however
            verbose : int: if 0 then just gives result, if 1 then it plots the clusters incl. test-set
                           cluster membership, if 2 it plots each x_test with its associated hashes
            *args : int: number of hyperplanes. if not provided it uses log2(len(x_train)/20) 
                         many hyperplanes (since each hyperplane divides the space into two, n hyperplanes
                         divide the space into 2^n that would mean we get about 20 elements per hash bin/hyperplane subspace)
                         
        '''
        self.k = k
        self.x_train = x_train  # shape(n_samples, dims)
        self.y_train = y_train
        self.dims = self.x_train.shape[1]
        self.method = method
        self.measure = self._euclid_dist if measure == 'euclid' else self._cosine_dist
        self.verbose = verbose
        self.hash_dict = {}
        if self.method == 'approximate':
            self.n_hyperplanes = args[0] if args else np.ceil(np.log2(len(self.x_train)/20)).astype(int)
            # generate the hyperplanes
            self.hyperplanes = np.random.randn(self.dims, self.n_hyperplanes)
            # get the sign of the dot product of hyperplanes and x_train
            hashes = np.sign(self.x_train @ self.hyperplanes) # shape(n_samples, self.n_hyperplanes)
            self.hash_dict = {tuple(array): [] for array in hashes}
            # append the indices of x_train to the hash dict using the hash as the dict key
            for i in range(len(hashes)):
                self.hash_dict[tuple(hashes[i])].append(i)

    def _cosine_dist(self, array1, array2, axis=None):
        return 1 - np.dot(np.squeeze(array1), np.squeeze(array2).T)/(np.linalg.norm(array1) * np.linalg.norm(array2))

    def _euclid_dist(self, array1, array2, axis=-1):
        return np.sqrt(np.sum((array1 - array2) ** 2, axis=axis))

    def _hamming_dist(self, array1, array2):
        # returns bit by bit difference betw. arrays
        return len(array1) - np.sum(array1==array2)

    def _measure_mtx(self, array1, array2, axis=-1):
        # numpy broadcasting proceeds from right to left
        # i.e. if x has shape(a, d), we're creating new axes so that
        # we have shapes s1 = (a, 1, d) and s2 = (1, b, d). with the axis=-1 parameter
        # in the measure function call, we calculate the distance/similarity (which is a scalar) along 
        # the rightmost dimension d and then broadcast the rest from right to left:
        # s2 has dim a whereas s1 just has dim 1, so s1 is broadcast to dim b;
        # after that the leftmost dim of s2 is broadcast to dim a, so that the result has shape(a, b)
        return self.measure(array1[:, None, :], array2[None, ...], axis=axis)

    def _plot(self, x_test, y_pred):
        assert self.dims == 2, 'plotting error! verify that data is 2D'
        num_classes = len(np.unique(self.y_train))
        color = cm.rainbow(np.linspace(0, 1, num_classes))
        plt.title(f'model output with number of classes {num_classes}')
        for k in range(num_classes):
            plt.scatter(x=self.x_train[self.y_train==k][:, 0], y=self.x_train[self.y_train==k][:, 1], 
                        marker='.', s=20, alpha=0.5, c=color[k])
            plt.scatter(x=x_test[y_pred==k][:, 0], y=x_test[y_pred==k][:, 1],
                        marker='+', s=250,  c=color[k])
        plt.show()

    def _subplot(self, x_test, closest_hash_ixes, pred):
        color = cm.rainbow(np.linspace(0, 1, len(x_test)))
        for i in range(len(x_test)):
            plt.title(f'subplot for:\nx_test {[i]} and its hashes. predicted class is {pred[i]}')
            plt.scatter(x=x_test[i, 0], y=x_test[i, 1], s=250, c='g', marker='+')
            plt.scatter(x=self.x_train[closest_hash_ixes[i]][:,0], y=self.x_train[closest_hash_ixes[i]][:,1],
            c=color[i], s=20, alpha=0.5, marker='.')
            plt.show()

    def predict(self,  x_test):
        if self.method == 'approximate':
            test_hash = np.sign(x_test @ self.hyperplanes) # shape(n_test, n_hyperplanes)
            closest_hash_ixes = [self.hash_dict.get(tuple(binary_row), []) for binary_row in test_hash]
            # if test_hash doesn't share a hyperplane divided subspace with hashed x_train
            # get closest x_train hash based on hamming distance and get at least k-many indices of x_train
            if any([len(hash_list) < self.k for hash_list in closest_hash_ixes]):
                for i in range(len(closest_hash_ixes)):
                    closest_hamming = sorted([(key, self._hamming_dist(test_hash[i], np.array(key))) for key in self.hash_dict.keys()], key=lambda x: x[1])
                    if closest_hash_ixes[i] == None or len(closest_hash_ixes[i]) <= self.k:
                        for ham in closest_hamming:
                            closest_hash_ixes[i] += self.hash_dict.get(ham[0]) if self.hash_dict.get(ham[0]) != None else []
                            if not closest_hash_ixes[i] or len(set(closest_hash_ixes[i])) >= self.k:
                                break
            closest_hash_ixes = [np.array(list(set(l))) for l in closest_hash_ixes]
            closest_ix_relative_to_hash_ixes = [np.argsort(self._measure_mtx(x_test[i, None], self.x_train[ix]))[:, :self.k] 
                                                         for i, ix in enumerate(closest_hash_ixes)]
            closest_ixes = np.vstack(np.squeeze([closest_hash_ixes[i][closest_ix_relative_to_hash_ixes[i]]
                            for i in range(len(closest_hash_ixes))]))
            labels = self.y_train[np.squeeze(closest_ixes)].astype(int)
            pred = np.array([np.argmax(np.bincount(label)) for label in labels])
            if self.verbose>=2:
                 self._subplot(x_test, closest_hash_ixes, pred)

        if self.method == 'exact':
            # get the indices of k closest points in x_train relative to x_test
            closest_ix = np.argsort(self._measure_mtx(x_test, self.x_train))[:, :self.k] # function return has shape(len(x_test), len(x_hash))
            # then use the indices to slice the labels of the train data
            labels = self.y_train[closest_ix].astype(int)
            pred = np.array([np.argmax(np.bincount(label)) for label in labels])
        if self.verbose == 0:
            return pred
        elif self.verbose >= 1:
            print(pred)
            self._plot(x_test, pred)
            return pred

