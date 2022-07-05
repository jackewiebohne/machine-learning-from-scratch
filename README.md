# machine-learning-from-scratch

## TOC:
1. [neural network](#neural-network)
2. [logistic regression](#logistic-regression)
3. [k cluster](#k-cluster)


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
k-means or k-medians clustering
can be combined with L1 (Manhattan) or L2 (Euclidean) distance metrics
has 4 different initializations: 
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
