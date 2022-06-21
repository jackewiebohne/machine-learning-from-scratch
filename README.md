# machine-learning-from-scratch-experiments

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


### logistic regression
