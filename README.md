# machine-learning-from-scratch-experiments

## logistic regression

## neural network:

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
