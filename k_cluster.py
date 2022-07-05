# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

class k_cluster():
    def __init__(self, k, method='means', distance='euclidian', init='bradley_fayyad', 
                 history=False, verbose=0, iter_threshold=200, convergence_memory=2):
        '''
        a clustering class that can implement a large variety of k-means, k-medoids etc, 
        in combination with an array of distance metrics and initialisation methods

        inputs:
            k : int: number of clusters
            method : str: 'means' for k-means, 'medians' for k-medians
            distance : str: 'euclidian' for L2 distance, 'manhattan' for L1 distance
            init : str: 'bradley_fayyad' for the eponymous method (most competitive and better than k-means++)
                        'k_means_pp' for k-means++, 'forgy' for randomly chosen datapoints as centroids,
                        'random_partition' for the means of k-many random partition means as centroids
            history : list: has length == n iterations and contains n tuples of:
                            (list(clustervalues), dict(cluster_ix: cluster_members))
            verbose: int: 0 prints nothing, 1 plots end result, 2 plots each iteration result
            iter_threshold: int: maximum number of iterations (in case of non-convergence)
            convergence_memory: int: number of iterations the outcomes has to not have changed to consider
                                     the algorithm fully converged
        '''
        self.k = k
        self.iterations = 0
        self.iter_threshold = iter_threshold
        self.history = history # will later be a list that contains iterations-many tuples(self.clustervals, self.clusters)
        self.verbose = verbose
        self.convergence_memory = convergence_memory
        self.clusters = {k:[] for k in range(self.k)}
        self.clustervals = [] # contains k many elements (i.e. the cluster centers based on the chosen method)
        self.func_dispatcher = {'euclidian': self._euclidian, 'manhattan': self._manhattan}
        self.method_dispatcher = {'means': self._means, 'medians': self._medians}
        self.init_dispatcher = {'forgy': self._forgy, 'random_partition': self._random_partition, 
                                'k_means_pp': self._k_means_pp, 'bradley_fayyad': self._bradley_fayyad}
        self.method = self.method_dispatcher[method]
        self.dist = self.func_dispatcher[distance]
        self.init = self.init_dispatcher[init]
        self.x = None
        self.dim = None
        self.n = None
        self.current_assignments = None # will contain a list of length n which gives the argmin wrt k using the chosen distance metric  
        self.initial_centers = None # since the k-clusters are highly sensitive to initialisation, we want to keep track of initial centers
        self.current_cumulative_distance = None # will contain the distances of datapoints to their cluster as a measure of the model's fit

    def predict(self, input):
        distances = []
        for k in range(self.k):
            distances.append(self.dist(self.clustervals[k], input))
        return np.argmin(distances, axis=0)

    def fit(self, x, centers=None):
        '''
        inputs:
            x : data to model (numpy array)
            centers (optional) : 
                if the model should not be initialised using the init method,
                but with given centers, then these need to be provided as list
        '''
        self.x = x
        self.n = x.shape[0]
        if len(x.shape) > 1: self.dim = x.shape[1] # features/dimensionality
        else: self.dim = 1
        assert(self.n >= self.k) # the number of clusters should be smaller than the number of observations
        if centers:
            self.clustervals = centers
        else:
            self.init()
        converged = 0
        while converged < self.convergence_memory and self.iterations <= self.iter_threshold:
            # print('clustervals',self.clustervals)
            # print('assignments', self.current_assignments)
            self.iterations += 1
            before = self.current_assignments
            self._assign_clusters()
            if self.iterations > 0 and np.all(before == self.current_assignments):
                converged += 1
            if self.verbose == 2: self._plot()
            if self.history: self.history.append((self.clustervals, self.clusters))
            self._update_clusters()
        if self.verbose == 1: self._plot()

    def _plot(self):
        assert self.dim == 2, 'plotting error! verify that data is 2D'
        color = cm.rainbow(np.linspace(0, 1, self.k)) # thanks to: https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
        plt.title(f'model at iteration {self.iterations}\
                    \ncumulative distance from centers:{self.current_cumulative_distance}')
        for k in range(self.k):
            plt.scatter(x=np.vstack(self.clusters[k])[:, 0], y=np.vstack(self.clusters[k])[:, 1], 
                        marker='.', s=20, alpha=0.4, c=color[k])
            plt.scatter(x=self.clustervals[k][0], y=self.clustervals[k][1], 
                        marker='+', s=200, c=color[k], alpha=1)
        plt.show()

    def _assign_clusters(self):
        self.clusters = {k:[] for k in range(self.k)}
        distances = [self.dist(self.clustervals[k], self.x) for k in range(self.k)]
        self.current_assignments = np.argmin(distances, axis=0)
        self.current_cumulative_distance = np.sum(np.column_stack(distances)[range(len(self.current_assignments)), self.current_assignments])
        k_indices = [np.where(self.current_assignments == k) for k in range(self.k)]
        self.clusters = {k: list(self.x[k_index]) for k, k_index in enumerate(k_indices)} # list splits matrix of shape (self.n, self.dim) into list, since that's what the other functions are expecting

    def _update_clusters(self):
        self.clustervals = [self.method(self.clusters[k]) for k in range(self.k)]

    def _means(self, x):
        return np.mean(x, axis=0)

    def _medians(self, x):
        return np.median(x, axis=0)

    def _euclidian(self, array1, array2, axis=1):
        return np.sum((array1 - array2) ** 2, axis=axis)

    def _manhattan(self, array1, array2, axis=1):
        return np.sum(np.abs(array1 - array2), axis=axis)

    def _k_means_pp(self):
        '''
        for the algorithm, see: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
        note that this algorithm is different from common implementations in blogs
        or websites (such as geeks for geeks). the original algorithm selects the new
        centroid probabilistically, not deterministically based on the maximum distance
        '''
        # choose the first point at random
        randint = np.random.randint(self.n, size=1)
        self.clustervals.append(self.x[randint])
        for k in range(self.k):
            # compute the initial distances (authors of original paper seem to imply
            # euclidian as the distance metric, but I'll also allow manhattan)
            # of dataset to most recently chosen clusterval
            # normalize (get probabilities) and choose a new center with that probability
            sq_distances = self.dist(self.x, self.clustervals[k]) ** 2
            total_distance = np.sum(sq_distances)
            probs = sq_distances/total_distance
            new_center_ix = np.random.choice(np.arange(self.n), size=1, p=probs)
            # let's make sure we don't choose two identical centervals
            while id(self.x[new_center_ix]) in map(id, self.clustervals):
                new_center_ix = np.random.choice(np.arange(self.n), size=1, p=probs)
            self.clustervals.append(self.x[new_center_ix])
        self.initial_centers = self.clustervals.copy()
        
    
    def _bradley_fayyad(self):
        '''
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.44.5872&rep=rep1&type=pdf
        creates subsamples, clusters those, then clusters the clusters, and of the meta-cluster solutions uses 
        the one that best fits the data as the initial clusters.

        Note that with large n and high dimensionality this is a computationally demanding initialisation. 
        it is designed for optimization of the kmeans objective. based on an extensive survey (https://arxiv.org/pdf/1209.1960.pdf)
        this is the most competitive method across a large array of datasets (synthetic and actual),
        and across a wide array of measures for goodness of fit. specifically it yields better optimizations than
        k-means++ which is the default init option in many implementations of k-means in various libraries.
        that being said the paper also outlines that despite the theoretically higher computational complexity
        of this initialisation, given that this init converges faster with better results, it might not be slower
        in practice (this also applies to k-means++)
        '''
        # create j small subsamples; I'll use 10% of the data points for the size of each subsample as in bradley & fayyad's example
        # and generate 10 subsamples; since each subsample has to be independent of the others, we have to do this iteratively
        subsamplesize = int(self.n * 0.1)
        J = 10
        assert subsamplesize, 'dataset is too small for this init method'
        assert subsamplesize > self.k, 'dataset is too small for this init method or k is too large'
        randints = np.hstack([np.random.randint(self.n, size=subsamplesize) for i in range(J)])
        # init the J submodels; we use _forgy; bradley & fayyad say nothing specific about what init method should be chosen
        # https://arxiv.org/pdf/1209.1960.pdf claim that bradley & fayyad use macqueen's 2nd method, which is what in our model is forgy
        CM = [k_cluster(self.k, method='means', distance='euclidian', init='forgy', 
                 history=False, verbose=0, iter_threshold=200, convergence_memory=1) for j in range(J)]
        for j in range(J):
            sample = self.x[randints[j*subsamplesize:(j+1)*subsamplesize]]
            # the following corresponds to KmeansMod in the paper 
            CM[j].fit(sample)
            maxit = 0 # some unlucky subsamples might only have have true clusters < k, so we limit the maximum iterations
            while not all(CM[j].clusters.values()) and maxit < 5: # if we have empty clusters
                maxit += 1
                empty_k = [k for k,v in CM[j].clusters.items() if not v] # get their indices
                # find datapoints that are furthest from their respective center
                furthest_pts = np.vstack([np.column_stack([self.dist(CM[j].clusters[k], CM[j].clustervals[k]), CM[j].clusters[k]]) for k in range(self.k) if k not in empty_k])
                furthest_pts = furthest_pts[furthest_pts[:, 0].argsort()[::-1]] # sort rows along 0-th column, which contains the distances, with biggest distance first
                # assign existing non-empty clustervals and reassign formerly empty clustervals as most distant datapoint for model fit method
                newcenters = []
                for k in range(self.k):
                    if CM[j].clusters.get(k, None):
                        newcenters.append(CM[j].clustervals[k]) 
                    else:
                        newcenters.append(furthest_pts[0, 1:])
                        furthest_pts = np.delete(furthest_pts, (0), axis=0)
                # run model again until we have a solution w/o empty clusters
                CM[j].fit(sample, newcenters)
        # init the clusters of the clusters; the forgy init will be overriden by assigning centers in the fit method
        FM = [k_cluster(self.k, method='means', distance='euclidian', init='forgy', 
                 history=False, verbose=0, iter_threshold=200, convergence_memory=1) for j in range(J)]
        best_distortion = float('inf')
        best_model = None
        for j in range(J):
            # use the clustervals as data for new clustering
            sample = np.vstack([CM[i].clustervals for i in range(J)])
            # override forgy init by assigning computed clustervals in CM[j] to the current model and run it
            FM[j].fit(sample, CM[j].clustervals)
            # now assign full dataset to the fit model
            FM[j].x = self.x
            # assign datapoints of full dataset to the clusters of model
            FM[j]._assign_clusters()
            # take model with lowest cumulative distance of full data from their respective centers
            if FM[j].current_cumulative_distance <= best_distortion:
                best_model = FM[j]
        # use the best init model for the full model initialization
        self.clustervals = best_model.clustervals
        self.initial_centers = self.clustervals.copy()

    def _forgy(self):
        # generate k-many random integers in range(n)
        randints = np.random.randint(self.n, size=self.k)
        # slice x based on randints and use them as clusters
        self.clustervals = [self.x[ix] for ix in randints] # we want this as a list instead of a matrix, which is what self.x[randints] would give
        self.initial_centers = self.clustervals.copy()

    def _random_partition(self):
        # generate n-many indices in range(k)
        sliceindices = np.arange(self.n) % self.k
        # assign each index (out of n indices) to cluster k
        for index, k in enumerate(sliceindices):
            self.clusters[k].append(self.x[index])
        # compute initial cluster values using the selected method
        self._update_clusters()
        self.initial_centers = self.clustervals.copy()
        # empty the clusters dict again
        self.clusters = {k:[] for k in range(self.k)}