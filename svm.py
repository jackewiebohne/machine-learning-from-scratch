import matplotlib.pyplot as plt
import numpy as np

# sheraton room reservation: #98772257
class SVM:
    def __init__(self, C=10, kernel='linear', coefficient=0, degree=3, gamma=1):
        '''
        soft margin svm (can be turned into hard margin by tuning gamma)
        note that this classifies binarily and any multiclass approaches would need to be 
        solved heuristically in one-vs-one or one-vs-rest schemes 
        params:
            C: float: weighting of regularisation loss (also often denoted as lambda, esp. in regression); higher C == harder margin
            kernel: str: linear (inner product), gaussian rbf, polynomial, None (i.e. no projection into higher dims)
            degree: int: degree of polynomial if kernel == 'polynomial'
            gamma: float: how many points to consider in rbf kernel, the higher gamma the fewer points will be considered and thus there'll be overfitting 
        '''
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coefficient = coefficient
        self.num_features, self.num_samples = None, None
        self.weights, self.bias = None, None
        self.initial_x, self.initial_y = None, None
        self.metrics = dict(accuracy=None, precision=None, recall=None, F1=None)

    def _linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def _polynomial_kernel(self, X1, X2):
        return (self.coefficient + np.dot(X1, X2.T)) ** self.degree

    def _rbf_kernel(self, X1, X2):
        if self.kernel == 'rbf' and self.gamma == None:
            # same as in sklearn
            gamma = 1 / (self.num_features * X1.var())
        else:
            gamma = 1.0 / (2 * self.gamma ** 2)
        # einsum is faster instead of: X_norm = np.sum(X ** 2, axis = -1)
        # see: https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python
        X1_norm = np.einsum('ij,ij->i', X1, X1)
        X2_norm = np.einsum('ij,ij->i', X2, X2) # this may seem redundant at train time but is necessary for testing
        # note that: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        distance_squared = X1_norm[:, None] + X2_norm[None,:] -2 * np.dot(X1, X2.T)
        return np.exp(-gamma * distance_squared)

    def _compute_kernel_matrix(self, X1, X2):
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == None:
            return X2
        else:
            raise ValueError("Invalid kernel specified.")

    def _calculate_loss(self, X, y):
        # correct
        margin = y * (np.dot(X, self.weights) + self.bias)
        hinge_loss = np.maximum(0, 1 - margin)
        regularization_loss = 0.5 * np.sum(self.weights ** 2) # einsum only works for weights > 1 D:  np.einsum('ij,ij->i', self.weights, self.weights)
        total_loss = np.mean(hinge_loss) + (self.C * regularization_loss) # c used like lambda or what is 1/C in other implementations
        return total_loss

    def _calculate_gradient(self, X, y):
        margin = y * (np.dot(X, self.weights) + self.bias)
        indicator = np.where(margin < 1, 1, 0)
        gradient_weights = -np.dot(X.T, y * indicator) / X.shape[0] + self.C * self.weights
        gradient_bias = -np.mean(y * indicator)
        return gradient_weights, gradient_bias

    def evaluate(self, x, y, save=True):
        '''
        params:
            x: np.array: inputs to evaluate on
            y: np.array: ground truths to compare predictions with
        '''
        # check again if these are calculated right
        preds = self.predict(x)
        TP = np.sum((preds == 1) & (y == 1)) # using "==" instead of "&" would give false results, since "&" only returns True iff both arrays are True and returns False in all other cases, whereas "==" returns True if both arrays evaluate to True OR False
        FP = np.sum((preds == 1) & (y == -1))
        FN = np.sum((preds == -1) & (y == 1))
        accuracy = (preds == y).sum()/len(y)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        F1 = 2 * (recall * precision) / (recall + precision)
        if save:
            self.metrics['accuracy'] = accuracy
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.metrics['F1'] = F1
        return dict(accuracy=accuracy, precision=precision, recall=recall, F1=F1)

    def _gradient_descent(self, X, y, learning_rate, num_epochs, early_stopping=1e-2):
        # implement batch sgd
        prev_loss = float('-inf')
        for epoch in range(num_epochs):
            gradient_weights, gradient_bias = self._calculate_gradient(X, y)
            self.weights -= learning_rate * gradient_weights
            self.bias -= learning_rate * gradient_bias
            if epoch%50==0:
                tmp_evaldict = self.evaluate(X, y, False)
                cur_loss = self._calculate_loss(X, y)
                print(f'current epoch: {epoch}, loss: {cur_loss}, metrics: {tmp_evaldict}')
                if early_stopping and np.abs(prev_loss-cur_loss) <= early_stopping:
                    print(f'early stopping criterion on loss fulfilled. current loss: {cur_loss}, change in loss: {np.abs(cur_loss - prev_loss)}, early stopping criterion: {early_stopping}')
                    break
                elif tmp_evaldict['accuracy'] == 1 or tmp_evaldict['F1'] == 1:
                    print(f'early stopping criterion on accuracy or F1 fulfilled. accuracy: {tmp_evaldict.get("accuracy")} and F1: {tmp_evaldict.get("F1")}')
                    self.metrics = tmp_evaldict
                    break
                prev_loss = cur_loss

    def fit(self, X, y, learning_rate=0.01, num_epochs=500, early_stopping=1e-3):
        self.num_samples, self.num_features = X.shape
        self.initial_x = X.copy()
        self.initial_y = y.copy()
        # Initialize weights and bias
        if self.kernel != None:
            self.weights = np.zeros(self.num_samples)
        else:
            self.weights = np.zeros(self.num_features)
        self.bias = 0.0
        # Compute kernel matrix
        kernel_matrix = self._compute_kernel_matrix(X, X)
        # Solve the optimization problem using gradient descent
        self._gradient_descent(kernel_matrix, y, learning_rate, num_epochs)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X[None,:]
        # https://stats.stackexchange.com/questions/299257/svm-how-to-get-predicted-output-from-svm-with-gaussian-rbf-kernel-andrew-ng
        # We need to use the original data set to transform the test data set
        # So, we need to use the original training set to transform and the equation is here that's K(xi,x),
        # where xi is the test data point and x is the original data set
        # if self.kernel:
        kernel_matrix = self._compute_kernel_matrix(self.initial_x, X).reshape(-1, self.weights.shape[0])
        predictions = np.sign(np.dot(kernel_matrix, self.weights) + self.bias)
        return predictions

    def get_support_vectors(self, threshold=1e-5):
        '''
        technically only gets the vectors closest to hyperplane
        '''
        kernel_matrix = self._compute_kernel_matrix(self.initial_x, X).reshape(-1, self.weights.shape[0])
        predictions = np.dot(kernel_matrix, self.weights) + self.bias
        neg_ix = np.where(predictions < 0, predictions, -np.inf).argmax()
        pos_ix = np.where(predictions > 0, predictions, np.inf).argmin()
        return self.initial_x[neg_ix], self.initial_x[pos_ix]


X = np.array([[-1, -1], 
              [-2, -1],
              [-3, -3],
              [-3, -2],
              [2, 3],
              [3, 3], 
              [1, 1], 
              [2, 1]])
X = np.vstack([-np.arange(10).reshape(5,2), np.arange(10).reshape(5,2) * 0.9, X]) 
y = np.array([1, -1, -1, -1, 1, 1, 1, 1])
y = np.hstack([-np.ones(5), np.ones(5), y])

svm = SVM(kernel=None)
svm.fit(X,y)
# print(svm.predict(np.array([-0.8, -1])))
# print(svm.predict(np.array([2,2])))

# Create a grid of points for plotting the hyperplane
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# print(Z)
# # Set the contour levels based on unique values in Z
levels = np.sort(np.unique(Z))
print(levels)


# Plot the hyperplane
plt.contour(xx, yy, Z, colors='k', levels=levels, alpha=0.5, linestyles=['-'])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
# support_vecs = np.vstack(svm.get_support_vectors())
# plt.scatter(support_vecs[:,0], support_vecs[:,1], c=[-2, 2])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Hyperplane')
plt.show()