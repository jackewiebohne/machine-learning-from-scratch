# machine-learning-from-scratch

## TOC:
1. [neural network](#neural-network)
2. [logistic regression](#logistic-regression)
3. [k cluster](#k-cluster)
4. [k nearest neighbor](#knn)
5. [decision tree (CART)](#decision-tree)
6. [random forest (based on CART trees)](#random-forest)


## neural network:

(the file contains all layers, activations etc. together. i'll split up into different files and clean up later)

  **layers:** 
  
  -  dense
  -  cnn (fully vectorised)
  -  dropout
    
   _forthcoming:_
  -    max and avg pooling
  -    layernorm
  -    flatten
  -    linear attention
  -    masked and multi-head attention blocks
  -    positional encoding
      
   **activations:**
   
  -    relu
  -    tanh
  -    sigmoid
  -    softmax
      
   _forthcoming:_
  -    selu
  -    leaky relu
  
   **optimizers**
  - adam
  - rmsprop
  - sgd
  - adagrad
 
   _forthcoming:_
  - adabelief
  - demonadam

   **losses**
   
  -    categorical crossentropy (joint with softmax as well as separate; can deal with both sparse and one-hot encodings)
  -    MSE
      
   **metrics**
   
  -    accuracy
      
   **callbacks**
  -    early stopping (on epoch end) based on accuracy in training, validation, or combination thereof
      
   **further features**
   
  -    prints and summarises the model in a manner similar to tensorflow
  -    also provides a history of loss, metrics etc.


_Example:_
```
xs = np.array([[1.0,2.0,3.0,4.0,5.0,6.0]]).T    
ys = np.array([[1.0,1.5,2.0,2.5,3.0,3.5]]).T
validation = np.array([[7.0, 8.0]]).T, np.array([[4.0, 4.5]]).T
m = Model([
            Dense(1) # simple linear regression
])
m.compile(xs.shape[1])
m.fit(xs,ys, epochs=120, optimizer=Adam(learning_rate=0.2), loss=MSE(), \
validation=validation, metrics=Accuracy(precision=0.2), \
verbose=1, callbacks=('train_valid_multiconditional', 0.99, 0.001))

m.predict(np.array([[9]])) # should be 5
```

or

```
m = Model([
            Conv2D(1, (3,3))
])
m.compile(n_features=3)
m.summary()
```
this gives

```
---------------------------
| conv2D_0 | (3, 3, 3, 1) |
---------------------------
Total trainable parameters: 10
```

(also included as a separate file called 'conv test.py' are various tests of my CNN backprop against other implementations)


## logistic regression

## k cluster
k-means or k-medians clustering that
can be combined with L1 (Manhattan) or L2 (Euclidean) distance metrics
and has 4 different initializations: 
 - forgy (random selection of k points from data)
 - random partition (creates k random partitions the data and uses their means/medians for initialization of centers)
 - bradly fayyad: (= for best results!) creates 10 subsamples of 10% of data for initial clusters, 
                  then clusters the clusters and uses the best fitting ones for initialization
 - k-means++: (can also be used with k-medians) the most commonly used, but not as competitive as bradley-fayyad

also has inbuilt plotting in choosable degrees of elaboration 

__Example:__
```
x1 = np.random.randn(200,2) 
x2 = np.random.randn(400,2) - 5
plt.scatter(x1[:,0], x1[:,1])
plt.scatter(x2[:,0], x2[:,1])
plt.show()
```
![](https://github.com/jackewiebohne/machine-learning-from-scratch/blob/master/k_cluster.png)
```
clusters = k_cluster(2, method='medians',init='forgy', verbose=2, distance='euclidian', convergence_memory=1)
clusters.fit(x)
print('initial centers', clusters.initial_centers)
print('best fit centers', clusters.clustervals, '\n')
```
this gives: 

![](https://github.com/jackewiebohne/machine-learning-from-scratch/blob/master/clusters.gif)
```
initial centers [array([-5.14219959, -5.25282999]), array([-6.79148878, -4.43687351])]
best fit centers [array([ 0.07997409, -0.02929862]), array([-5.10684104, -4.96894499])] 
```

## knn
k nearest neighbor
can do 'exact' and 'approximate' (via locality sensitive hashing, using hyperplanes and hamming distance)
exact will often be faster (since it is fully vectorised) unless dealing with a huge dataset
default distance measure is euclidian, but cosine distance is also implemented (but not recommended)
includes plotting at choosable levels of elaboration

__Example:__
```
x1 = np.random.randn(200,2)
x2 = np.random.randn(400,2) - 5
y1 = np.ones(200)
y2 = np.zeros(400)
x3 = np.random.randn(10, 2) - 3
plt.scatter(x1[:,0], x1[:,1])
plt.scatter(x2[:,0], x2[:,1])
plt.scatter(x3[:,0], x3[:,1])
plt.show()
```
shows:

![](https://github.com/jackewiebohne/machine-learning-from-scratch/blob/master/knn.png) 
```
kn = knn(3,x_train = np.vstack((x1, x2)), y_train=np.hstack((y1, y2)), method='approximate', measure='euclid', verbose=1)
out = kn.predict(x3)
```
outputs:

![](https://github.com/jackewiebohne/machine-learning-from-scratch/blob/master/knn%20pred.png)
```
[0 1 0 0 0 0 0 0 1 1] # 'exact' would output [0 0 0 0 0 0 0 0 1 1]
```


## decision tree
based on Breiman's CART algorithm
it can regress and classify, but does not handle categorical data

__Example:__
```
########### Comparing Classification and Regression with Scikit-learn ###########
# examples taken and amended from: https://github.com/zziz/cart/blob/master/cart.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree as sktree

# Classification Tree
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
    
# Regression
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
```


## random forest
random forest based on Breiman's random forest, which is in turn based on his CART decision trees

__Example:__
```
# more or less same example as for decision trees above
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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
