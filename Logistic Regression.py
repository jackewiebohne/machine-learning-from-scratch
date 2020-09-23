import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt


class LogisticRegression():
    def __init__(self, X, y, regularisation_param=0, max_iter=1000): #make sure that the inputs are arrays
        self.X = np.insert(X, 0, 1, axis=1) #inserting ones into first column of training set so that matrix multiplication with theta is possible for the intercept theta_0
        self.y = y
        # self.k = klasses
        self.lmbda = regularisation_param
        self.m = X.shape[0] #gives the length of the rows, i.e. the training samples
        self.n = np.insert(X, 0, 1, axis=1).shape[1] #gives the length of columns, i.e. the features
        self.max_iter = max_iter
        self.theta = np.zeros(self.n,) #initialising theta with all zeros of the size of all features (since theta will be multiplied by the features)


    def Sigmoid(self, z):
        return (1/(1 + np.exp(-(z))))


    def GradientDescent(self, learning_rate=0.01, GD_batch_size=100):
        alpha = learning_rate
        thetas = []
        grads = []
        iterations = []
        theta_shifts = []

        if GD_batch_size != self.m:

            for iteration in range(self.max_iter):

                theta_shift = np.concatenate(([0], self.theta[1:])) # since the intercept is not regularised we multiply the regularisation parameter lmbda with theta_shift with a 0 inserted at the position of theta_0/the intercept
                index = random.randint(0, (self.m - (GD_batch_size) ) ) # generate a random number that will be used as index for slicing the training examples m # the subtraction is to avoid running out of range, this random number+GD_batch_size will be the slicing scope of the training samples
                h_batch = self.Sigmoid(self.X[index : index + GD_batch_size] @ self.theta)
                reg = (self.lmbda/GD_batch_size) * theta_shift # regularisation
                grad = ( (1/GD_batch_size) * ( ( (h_batch - self.y[index : index + GD_batch_size]) @ self.X[index : index + GD_batch_size]) ) ) + reg
                self.theta = self.theta - (alpha * grad)

                #appending the terms
                thetas.append(self.theta)
                theta_shifts.append(theta_shift)
                grads.append(grad)
                iterations.append(iteration)

        else:

            for iteration in range(self.max_iter):

                theta_shift = np.concatenate(([0], self.theta[1:]))
                h = self.Sigmoid(self.X @ self.theta)
                reg = (self.lmbda/self.m) * theta_shift
                grad = ( (1/self.m) * ( (h - self.y) @ self.X) ) + reg
                self.theta = self.theta - (alpha * grad)

                #appending the terms
                thetas.append(self.theta)
                theta_shifts.append(theta_shift)
                grads.append(grad)
                iterations.append(iteration)

        return thetas, theta_shifts, grads, iterations


    def CostFunction(self, learning_rate=0.01, GD_batch_size=100):
        thetas, theta_shifts, grads, iterations = self.GradientDescent(learning_rate, GD_batch_size)
        costs = []

        for iteration in range(self.max_iter):

            h = self.Sigmoid(self.X @ thetas[iteration])
            reg = (self.lmbda/(2 * self.m) ) * (theta_shifts[iteration] @ theta_shifts[iteration])
            J = (1/self.m) * ( (-self.y).T @ np.log(h) - (1 - self.y).T @ np.log(1 - h) ) + reg

            costs.append(J)

        return costs, grads, iterations


    def Predict(self):
        thetas, theta_shifts, grads, iterations = self.GradientDescent(learning_rate=0.01, GD_batch_size=100)
        prediction = self.Sigmoid(self.X @ thetas[-1])
        return prediction


x = [] # e.g.: df.iloc[:,:-1].to_numpy()
y = [] # e.g.: df.iloc[:, -1].to_numpy()



l = LogisticRegression(x,y, max_iter=100, regularisation_param=0)
# thetas, theta_shifts, grads, iterations = l.GradientDescent(GD_batch_size=100, learning_rate=0.001)
costs, grads, iterations = l.CostFunction(learning_rate=0.0001, GD_batch_size=len(y))



##graph cost and gradient per iteration
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(iterations, grads)
ax2.plot(iterations, costs)

ax1.set_xlabel('iterations')
ax2.set_xlabel('iterations')
ax1.set_ylabel('gradients')
ax2.set_ylabel('costs')

plt.show()

