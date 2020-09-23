import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt


df = pd.read_csv('C:/Users/jackewiebohne/Documents/studium/ML Stanford/machine-learning-ex2/ex2/ex2data1.txt', sep=',', names=['exam 1', 'exam 2', 'admitted'])

# a = np.array([1, 2 ,3])
# b = np.array([2, 2, 2])
# c = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
# d = np.array([[1,1, 1], [2,2,2], [3,3,3]])
# e = np.insert(c, 0, 1, axis=1) # 1st input is the vector/array to be inserted into; 2nd input is the position to be inserted into (here: first); 3rd is what is to be inserted (here:1); 4th is what axis (here: column) => i.e. 1 is inserted as the first element of the column
# print(e)
# print(c@b) #matrix multiplication
# print(np.shape(c)) # outputs rows x columns
# print(a + b) #element-wise addition
# print(a*b) #element-wise multiplication
# print(c.T) #transposes matrix/array
# print(np.invert(a)) #inverts array/matrix
# print(len(c)) #outputs column number
# print(np.size(a)) #outputs row number
# print(d)
# a[:,0] = np.array[0, 0, 0]

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

                theta_shift = np.concatenate(([0], self.theta[1:])) # concantenate takes a tuple # since the intercept is not regularised we multiply the regularisation parameter lmbda with theta_shift with a 0 inserted at the position of theta_0/the intercept
                index = random.randint(0, (self.m - (GD_batch_size) ) ) # generate a random number that will be used as index for slicing the training examples m # the subtraction is to avoid running out of range, this random number+GD_batch_size will be the slicing scope of the training samples
                h_batch = self.Sigmoid(self.X[index : index + GD_batch_size] @ self.theta)
                reg = (self.lmbda/GD_batch_size) * theta_shift # regularisation
                grad = ( (1/GD_batch_size) * ( ( (h_batch - self.y[index : index + GD_batch_size]) @ self.X[index : index + GD_batch_size]) ) ) + reg
                self.theta = self.theta - (alpha * grad)
                # print(grad)

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
                # print(grad)

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


x = df.iloc[:,:-1].to_numpy()
y = df.iloc[:, -1].to_numpy()



l = LogisticRegression(x,y, max_iter=100, regularisation_param=0)
# thetas, theta_shifts, grads, iterations = l.GradientDescent(GD_batch_size=100, learning_rate=0.001)
costs, grads, iterations = l.CostFunction(learning_rate=0.0001, GD_batch_size=len(y))


# x1split = []
# x2split = []
# for lst in x:
#     x1split.append(lst[0])
#     x2split.append(lst[1])

##graph cost and gradient per iteration
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# fig, axs = plt.subplots(1,2)
ax1.plot(iterations, grads)
ax2.plot(iterations, costs)

ax1.set_xlabel('iterations')
ax2.set_xlabel('iterations')
ax1.set_ylabel('gradients')
ax2.set_ylabel('costs')

# y_1 = []
# y_0 = []
# pos_costs = []
# neg_costs = []
# for i in y:
#     if i == 1:
#         y_1.append(y[i])
#         pos_costs.append(costs[i])
#     else:
#         y_0.append(y[i])
#         neg_costs.append(costs[i])
#
# print(len(y_0), len(y_1), len(pos_costs) ,len(neg_costs))


###mapping the cost function###
# thetas = 1
# X = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
# y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
# costs = []
# hs = []
# for i in range(9):
#
#     h = 1/(1 + np.exp(-(X[i] * thetas)))
#     hs.append(h)
#     J = (1/1) * ( (-y[i]).T * np.log(h) - (1 - y[i]).T * np.log(1 - h) )
#     costs.append(J)
# plt.scatter(hs, costs, c=y)
# plt.xlabel('h')
# plt.ylabel('costs')
###

# plt.scatter(iterations, grads)
# plt.xlabel('iterations')
# plt.ylabel('gradients')
# plt.title('gradient per iteration')
plt.show()

