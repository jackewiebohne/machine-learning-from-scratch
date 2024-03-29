# -*- coding: utf-8 -*-
import numpy as np
import math
from abc import ABC, abstractmethod
from time import time
from matplotlib import pyplot as plt
import sys

# #install python 3.9
# !sudo apt-get update -y
# !sudo apt-get install python3.9

# #change alternatives
# !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
# !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

# !python --version

class Layer(ABC):
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backward(self):
        pass
    @abstractmethod
    def predict(self):
        pass
    def __str__(self):
        methods = '\n    '.join([m for m in dir(self) if not str(m).startswith('_')]) # this is b/c backslashes are not allowed inside curly brackets
        return f'layer name: {self.name}; layer type: {self.l_type}\nmethods:\n    {methods}'

class Loss(ABC):
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backward(self):
        pass
    def __str__(self):
        methods = '\n    '.join([m for m in dir(self) if not str(m).startswith('_')])
        return f'layer name: {self.name}; layer type: {self.l_type}\nmethods:\n    {methods}'

class Optimizer(ABC):
    pass

class Metric(ABC):
    def __str__(self):
        return f'{self.name}'

class Conv2D(Layer):
    l_type = 'conv2D'
    def __init__(self, filter_num, filter_sizes=(3,3), stride=1, pad_size=None, pad_val=0, float_precision=np.float32):
            '''
            inputs: 
               filter_num: int: number of filters/filter_depth
               filter_sizes: tuple(int): tuple of ints for filter height and width

            expects input tensors to be of shape: [batchsize, height, width, channels]
            after init, filter_shapes are: [fh, fw, channel in, channel out] 
            '''
            self.Cout = filter_num
            self.fh, self.fw = filter_sizes
            self.pad_size = pad_size
            if self.pad_size == None: self.pad_size = (self.fh-1)//2
            self.num_trainable_params = self.fh * self.fw * self.Cout + 1
            self.stride = stride
            self.pad_val = pad_val
            self.float_precision = float_precision
            self.prev_l = None
            self.next_l = None
            self.inputs = None # this will save the strided windows, not the original input
            self.input_shape = None
            self.Wout = None
            self.Hout = None
            self.w = None
            self.b = None
            self.out = None
            self.dw = None
            self.db = None
            self.dout = None
            self.trainable = True
            self.name = None

    def init_weights(self, n_features):
        # only basic init_type implemented
        # for float precision, see http://proceedings.mlr.press/v37/gupta15.pdf
        # stochastic rounding is not implemented here, however
        self.w = np.random.randn(self.fh, self.fw, n_features, self.Cout) / np.sqrt(n_features/2).astype(self.float_precision)
        self.b = np.zeros((1, self.Cout)).astype(self.float_precision)

    def _pad(self, array, pad_size, pad_val):
        '''
        only symmetric padding is possible
        ''' 
        return np.pad(array, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', constant_values=(pad_val, pad_val))

    def _dilate(self, array, stride_size, pad_size, symmetric_filter_shape, output_image_size):
        # on dilation for backprop with stride>1, see: https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
        # see also: https://leimao.github.io/blog/Transposed-Convolution-As-Convolution/
        pad_bottom_and_right = (output_image_size + 2 * pad_size - symmetric_filter_shape) % stride_size
        for m in range(stride_size - 1):
            array = np.insert(array, range(1, array.shape[1], m + 1), 0, axis=1)
            array = np.insert(array, range(1, array.shape[2], m + 1), 0, axis=2)
        for _ in range(pad_bottom_and_right):
            array = np.insert(array, array.shape[1], 0, axis=1)
            array = np.insert(array, array.shape[2], 0, axis=2)
        return array


    def _windows(self, array, stride_size, filter_shapes, out_height, out_width):
        '''
        inputs:
            array to create windows of
            stride_size: int
            filter_shapes: tuple(int): tuple of filter height and width
            out_height and out_width: int, respectively: output sizes for the windows
        returns:
            windows of array with shape (excl. dilation): 
                array.shape[0], out_height, out_width, filter_shapes[0], filter_shapes[1], array.shape[3]
        '''            
        strides = (array.strides[0], array.strides[1] * stride_size, array.strides[2] * stride_size, array.strides[1], array.strides[2], array.strides[3])
        return np.lib.stride_tricks.as_strided(array, shape=(array.shape[0], out_height, out_width, filter_shapes[0], filter_shapes[1], array.shape[3]), strides=strides, writeable=False)


    def forward(self, x):
        '''
        expects inputs to be of shape: [batchsize, height, width, channels]
        after init, filter_shapes are: [fh, fw, channel in, channel out] 
        '''
        self.input_shape = x.shape
        x_pad = self._pad(x, self.pad_size, self.pad_val).astype(self.float_precision) # cast all values to float32 to gain computational speed and save memory at minimal accuracy loss (np standard is float64); float. pt. accuracy could even be reduced more, see https://arxiv.org/pdf/1502.02551.pdf
        # get the shapes
        batch_size, h, w, Cin = self.input_shape
        # calculate output sizes; only symmetric padding is possible
        self.Hout = (h + 2*self.pad_size - self.fh) // self.stride + 1
        self.Wout = (w + 2*self.pad_size - self.fw) // self.stride + 1
        x_windows = self._windows(array=x_pad, stride_size=self.stride, filter_shapes=(self.fh, self.fw),
                        out_width=self.Wout, out_height=self.Hout) # 2D matrix with shape (batch_size, Hout, Wout, fh, fw, Cin)
        self.out = np.tensordot(x_windows, self.w, axes=([3,4,5], [0,1,2])) + self.b
        self.inputs = x_windows
        ## alternative 1: einsum approach, slower than other alternatives
        # self.out = np.einsum('noufvc,fvck->nouk', x_windows, self.w) + self.b
        ## alternative 2: column approach with simple dot product
        # z = x_windows.reshape(-1, self.fh * self.fw * Cin) @ self.W.reshape(self.fh*self.fw*Cin, Cout) + self.b # 2D matrix with shape (batch_size * Hout * Wout, Cout)
        # self.dout = z.reshape(batch_size, Hout, Wout, Cout)

    def backward(self,dout_descendant):
        '''
        dout_descendant has shape (batch_size, Hout, Wout, Cout)
        ''' 
        # get shapes
        batch_size, h, w, Cin = self.input_shape
        # we want to sum everything but the filters for b
        self.db = np.sum(dout_descendant, axis=(0,1,2), keepdims=True) # shape (1,1,1, Cout)
        # for dW we'll use the column approach with ordinary dot product
        dout_descendant_flat = dout_descendant.reshape(-1, self.Cout) # new shape (batch_size * Hout * Wout, Cout)
        x_flat = self.inputs.reshape(-1, self.fh * self.fw * Cin) # shape (batch_size * Hout * Wout, fh * fw * Cin)
        dw = x_flat.T @ dout_descendant_flat # shape (fh * fw * Cin, Cout)
        self.dw = dw.reshape(self.fh, self.fw, Cin, self.Cout)
        del dout_descendant_flat # free memory
        # for dinputs, things are complicated. we'll get padded and dilated windows of dout_descendant and perform the tensordot with 180 rotated W
        # for details, see https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
        # also: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c ; also: https://youtu.be/Lakz2MoHy6o?t=835
        Wrot180 = np.rot90(self.w, 2, axes=(0,1)) # or also self.w[::-1, ::-1, :, :]
        # backprop for forward with stride > 1 is done on windowed dout that's padded and dilated with stride 1
        dout_descendant = self._dilate(dout_descendant, stride_size=self.stride, pad_size=self.pad_size, 
                                       symmetric_filter_shape=self.fh, output_image_size=h)
        dout_descendant = self._pad(dout_descendant, pad_size=self.fw-1, pad_val=self.pad_val) # pad dout_descendant to dim: fh-1 (or fw-1); only symmetrical filters are supported
        dout_descendant = self._windows(array=dout_descendant, stride_size=1, filter_shapes=(self.fh, self.fw),
                            out_height=(h + 2 * self.pad_size), out_width=(w + 2 * self.pad_size)) # shape: (batch_size * h_padded * w_padded, fh * fw * Cout)
        self.dout = np.tensordot(dout_descendant, Wrot180, axes=([3,4,5],[0,1,3]))
        self.dout = self.dout[:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size, :]
        ## einsum alternative, but slower:
        # dinput = np.einsum('nhwfvk,fvck->nhwc', dout_windows, self.W)

    def predict(self, x):
        '''
        expects inputs to be of shape: [batchsize, height, width, channels]
        after init, filter_shapes are: [fh, fw, channel in, channel out] 
        '''
        x_pad = self._pad(x, self.pad_size, self.pad_val).astype(self.float_precision)
        batch_size, h, w, Cin = self.input_shape
        x_windows = self._windows(array=x_pad, stride_size=self.stride, filter_shapes=(self.fh, self.fw),
                        out_width=self.Wout, out_height=self.Hout) # 2D matrix with shape (batch_size, Hout, Wout, fh, fw, Cin)
        return np.tensordot(x_windows, self.w, axes=([3,4,5], [0,1,2])) + self.b
        
class Flatten(Layer):
    pass

class MaxPool(Layer):
    pass

class Conv1D(Layer):
    # could use np.convolve # only works with 1D
    pass

class Layernorm(Layer):
    pass

class Callback:
    def __init__(self, method):
        '''
        inputs:
            method: str or tuple(str, float) or tuple(str, float, float):
                    for the string the options are:
                        - 'train_accuracy': stop when a certain accuracy level in training data
                            is reached (default 0.9)
                        - 'validation_accuracy': same as above for validation (default 0.9)
                        - 'train_valid_diff': stop when the difference between train and validation
                            accuracy is below a threshold (default 0.05); note that if train and validation
                            accuracy are below the threshold and not 0, this doesn't necessarily mean that
                            they have high accuracy (if e.g. the train and validation datasets are really small
                            there are unlucky moments when, say, both train and validation accuracy are at 1/3%
                            which would trigger this condition, so treat with caution)
                        - 'train_valid_multiconditional': if all three above conditions are met (either with their
                            default thresholds or with the thresholds provided as floats in the tuple)
                        - 'loss': if the loss function fall below a threshold. there is no default threshold here,
                            so it will need to be provided as a float in a tuple together with str

                    for the floats in the tuple the options are:
                        - any float ranging from 0 to 1; if multiple floats are provided they are ignored, unless
                            the str is 'train_valid_multiconditional', in which case the first float will be used
                            for the threshold in train and validation accuracy threshold and the second for their 
                            difference threshold

                    note that any number of values above three in the tuple or list is ignored
        '''
        if type(method) == str:
            self.method, self.thresholds = method, None
        elif type(method) == tuple:
            self.method = method[0]
            self.thresholds = [*method[1:]]

    def __call__(self, history):
        if self.method == 'train_accuracy':
            if self.thresholds:
                return self.train_accuracy(history, self.threshold[0])
            else:
                return self.train_accuracy(history)
        elif self.method == 'validation_accuracy':
            if self.thresholds:
                return self.valid_accuracy(history, self.thresholds[0])
            else:
                return self.valid_accuracy(history)
        elif self.method == 'train_valid_diff':
            if self.thresholds:
                return self.train_valid_diff(history, self.thresholds[0])
            else:
                return self.train_valid_diff(history)
        elif self.method == 'train_valid_multiconditional':
            if self.thresholds:
                return self.train_valid_multiconditional(history)
            else:
                return self.train_valid_multiconditional(history)
        elif self.method == 'loss':
            if self.thresholds:
                return self.loss(history, self.thresholds[0])
            else:
                raise ValueError('when using callbacks with loss a threshold must be provided, no threshold default value exists')
        else:
            raise ValueError('callback method not recognized')

    def train_accuracy(self, history, threshold=0.9):
        return history['train_accuracy'][-1] >= threshold 

    def valid_accuracy(self, history, threshold=0.9):
        return history['validation_accuracy'][-1] >= threshold 
    
    def train_valid_diff(self, history, threshold=0.05):
        trainacc = history['train_accuracy'][-1]
        validacc = history['validation_accuracy'][-1]
        if trainacc != 0 and validacc != 0:
            return abs(trainacc - validacc) <= threshold
    
    def train_valid_multiconditional(self, history, *args):
        if args:
            if self.train_valid_diff(history, args[1]) and self.valid_accuracy(history, args[0]) and self.train_accuracy(history, args[0]):
                return True
        elif self.train_valid_diff(history) and self.valid_accuracy(history) and self.train_accuracy(history):
            return True

    def loss(self, history, threshold):
        return history['loss'][-1] >= threshold

class Dense(Layer):
    # this layer expects 2D arrays!
    l_type = 'dense'
    def __init__(self, n):
        # inputs: 
        #   n: number of neurons
        self.n = n
        self.prev_l = None
        self.next_l = None
        self.inputs = None
        self.w = None
        self.b = None
        self.out = None
        self.dw = None
        self.db = None
        self.dout = None
        self.trainable = True
        self.num_trainable_params = self.n + 1
        self.name = None
    
    def init_weights(self, n_features, init_type='xavier'):
        '''
        for init_type only 'xavier' and 'he' are available, default: xavier
        '''
        if init_type == 'he':
            # use he initialisation if next layer is relu
            self.w = np.random.randn(n_features, self.n) / np.sqrt(2/n_features)
            self.b = np.zeros((1, self.n))
        else:
            # using xavier for all other activations
            self.w = np.random.randn(n_features, self.n) / np.sqrt(1/n_features)
            self.b = np.zeros((1, self.n))

    def forward(self, inputs):
        self.inputs = inputs
        self.out = inputs @ self.w + self.b

    def backward(self, dout_descendant):
        # inputs:
        #     dout_descendant: the gradient of the next layer (most likely: activation/loss)
        self.dout = np.dot(dout_descendant, self.w.T) 
        self.dw = np.dot(self.inputs.T, dout_descendant)
        self.db = np.sum(dout_descendant, axis=0, keepdims=True) # keepdims to prevent shape reduction

    def predict(self, inputs):
        return inputs @ self.w + self.b

    def __str__(self):
        inherited = f'layer shapes: {self.w.shape}, neurons: {self.n}\n' if self.w else f'no final shape yet, neurons: {self.n}\n'
        inherited += super().__str__()
        return inherited

class Softmax(Layer):
    l_type = 'softmax'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.out = None
        self.dout = None
        self.num_trainable_params = 0

    def forward(self, inputs):
        # inputs is previous layers' dout
        exp = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True)) # -np.max() is for numerical stability
        self.out = exp/np.sum(exp, axis=-1, keepdims=True)

    def predict(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        return np.argmax(exp/np.sum(exp, axis=-1, keepdims=True), axis=-1)

    def backward(self, input):
        # inputs are the derivatives of next layer
        # if i == j
        ij = np.einsum('ij,jk->ijk', self.out, np.eye(input.shape[-1])) # creates a diagonal matrix over last dim, equivalent to: h_i * (1 − h_j), see: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
        # if i != j
        inonj = np.einsum('ij,ik->ijk', self.out, self.out) # equivalent to: - h_i * h_j (outer product over indices i, j)
        joint_jacobian = ij - inonj # shape: (samples, features, features)
        self.dout = np.einsum('ijk,ik->ij', joint_jacobian, input)
        ####
        # # above is vectorised version for:
        # lst = []
        # for index, (single_output, single_dvalues) in enumerate(zip(self.out, inputs)):
        #     single_output = single_output.reshape(-1, 1) 
        #     jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
        #     lst.append(np.dot(jacobian_matrix, single_dvalues))
        # self.dout = np.array(lst)

class Sm_Crossentropy(Loss):
    # softmax and crossentropy combined for more computational efficiency
    l_type = 'softmax_crossentropy'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.out = None
        self.dout = None
        self.softmax = Softmax()
        self.loss = Crossentropy()
    
    def forward(self, pred, y):
        self.softmax.forward(pred)
        self.out = self.loss.forward(self.softmax.out, y)
    
    def backward(self, pred, y):
        if len(y.shape) > 1:
            y = np.argmax(y, axis=-1)
        self.dout = self.softmax.out.copy() # self.softmax.out == predictions whose output was already calculated and saved; copy them for safe modification
        self.dout[range(y.shape[0]), y] -= 1 # since the joint derivative is pred - y
        print(self.dout)
        self.dout /= y.shape[0]

class Crossentropy(Loss):
    l_type = 'crossentropy'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.out = None
        self.dout = None
    
    def forward(self, pred, y):
        # if y sparse, i.e. 0, 1, 2, ..., k for k classes
        y = np.squeeze(y)
        if len(y.shape) == 1:
            correct_confidences = pred[range(y.shape[0]), y] # slices the softmax confidence for the correct class, i.e. the k-th column for each row
        # if y one-hot encoded
        else:
            correct_confidences = np.sum(pred * y, axis=-1)
        correct_confidences = np.clip(correct_confidences, 1e-7, 1 - 1e-7) # clip the min and max vals to be slightly higher than 0 and slightly less than 1, respectively
        log_likelihood = - np.log(correct_confidences)
        self.dout = np.mean(log_likelihood)
    
    def backward(self, pred, y):
        y = np.squeeze(y)
        # if y sparse
        if len(y.shape) == 1:
            y = np.eye(pred.shape[1])[y] # one-hot encode classes; alternative to pred.shape[1] is max val of y
        self.dout = - (y / pred) * (1 / y.shape[0])

class Relu(Layer):
    l_type = 'relu'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.out = None
        self.dout = None
        self.num_trainable_params = 0

    def forward(self, inputs):
        self.out = np.maximum(0, inputs)
    
    def backward(self, inputs):
        # inputs are the derivatives of prev layer wrt to its input
        self.dout = (inputs > 0) * 1 # the multiplication by 1 turns the boolean outcome of (x>0) into 1s and 0s

    def predict(self, inputs):
        return np.maximum(0, inputs)

class Tanh(Layer):
    l_type = 'tanh'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.out = None
        self.dout = None
        self.num_trainable_params = 0

    def forward(self, inputs):
        self.out = np.tanh(inputs)

    def backward(self, inputs):
        # inputs are the derivatives of prev layer wrt to its input, i.e. dout
        self.dout = 1-(np.tanh(inputs)**2)

    def predict(self, inputs):
        return np.tanh(inputs)

class Sigmoid(Layer):
    l_type = 'sigmoid'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.out = None
        self.dout = None
        self.num_trainable_params = 0

    def forward(self, inputs):
        self.out = 1/(1 + np.exp(-inputs))

    def backward(self, inputs):
        # inputs are the derivatives of prev layer wrt to its input
        _sigmoid = 1/(1 + np.exp(-inputs))
        self.dout = _sigmoid * (1 - _sigmoid)

    def predict(self, inputs):
        return 1/(1 + np.exp(-inputs))

class Accuracy(Metric):
    l_type = 'metric'
    def __init__(self, precision=0.05):
        self.name = 'accuracy'
        self.precision = precision
    def __call__(self, pred, y, l_prev=None):
        # precision is for non-categorical comparison; 
        #   if the difference between y and pred is larger than precision
        #   then it will count as an error
        if 'softmax' in l_prev:
            return (np.argmax(pred, axis=-1) == y).mean()
        elif 'sigmoid' in l_prev:
            return ((pred > 0.5) == y).mean()
        else:
            # print('numerical accuracy', y, pred)
            return ((np.abs(y-pred) < self.precision) * 1).mean()

class Dropout(Layer):
    l_type = 'dropout'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.mask = None
        self.out = None
        self.num_trainable_params = 0

    def forward(self, inputs, p=0.4):
        # can this be done without inputs at compile time?
        self.mask = np.random.choice([0,1], size=inputs.shape, p=[p, (1-p)]) / (1-p) 
        self.out = self.mask * inputs

    def backward(self, dinputs):
        self.dout = self.mask * dinputs

    def predict(self, inputs):
        return inputs # no droupout at prediction/testing

class MSE(Loss):
    l_type = 'mse'
    def __init__(self):
        self.name = None
        self.prev_l = None
        self.next_l = None
        self.out = None
        self.dout = None

    def forward(self, pred, y):
        # pred: predicted; y: ground truth
        self.out = np.mean((y-pred)**2)

    def backward(self, pred, y):
        # pred: predicted; y: ground truth
        self.dout = -2/y.shape[0] * (y - pred)

# optimisers
class SGD(Optimizer):
    l_type = 'sdg'
    def __init__(self, learning_rate=0.001, momentum=0):
        self.lr = learning_rate
        self.momentum = momentum
        self.w_changes = {}
        self.b_changes = {}

    def __call__(self, current_epoch, trainable_layer):
        if self.momentum:
            if self.w_changes.get(trainable_layer.name, None) == None:
                self.w_changes[trainable_layer.name] = np.zeros_like(trainable_layer.w)
                self.b_changes[trainable_layer.name] = np.zeros_like(trainable_layer.b)
                
            w_change = self.momentum * self.w_changes[trainable_layer.name] + trainable_layer.dw * self.lr
            b_change = self.momentum * self.b_changes[trainable_layer.name] + trainable_layer.db * self.lr
            trainable_layer.b -= b_change
            trainable_layer.w -= w_change
            self.w_changes[trainable_layer.name] = w_change
            self.b_changes[trainable_layer.name] = b_change   
                
        else:
            trainable_layer.b -= self.lr * self.trainable_layer.db
            trainable_layer.w -= self.lr * self.trainable_layer.dw

class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0., epsilon=1e-7):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.decay = decay

    def __call__(self, current_epoch, trainable_layer):
        w_change = trainable_layer.dw ** 2
        b_change = trainable_layer.db ** 2
        if self.decay:
            self.lr *= (1 / (1 * self.decay * current_epoch))
        trainable_layer.w -= self.lr * trainable_layer.dw / np.sqrt(w_change + self.epsilon)
        trainable_layer.b -= self.lr * trainable_layer.db / np.sqrt(b_change + self.epsilon)


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0.99, epsilon=1e-7):
        self.lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.w_divisors = {}
        self.b_divisors = {}

    def __call__(self, current_epoch, trainable_layer):
        if self.w_divisors.get(trainable_layer.name, None) == None:
            self.w_divisors[trainable_layer.name] = np.zeros_like(trainable_layer.w)
            self.b_divisors[trainable_layer.name] = np.zeros_like(trainable_layer.b)

        w_divisor = self.decay * self.w_divisors[trainable_layer.name] + (1 - self.decay) * trainable_layer.dw**2
        b_divisor = self.decay * self.b_divisors[trainable_layer.name] + (1 - self.decay) * trainable_layer.db**2
        trainable_layer.w -= self.lr * trainable_layer.dw / (np.sqrt(w_divisor) + self.epsilon)
        trainable_layer.b -= self.lr * trainable_layer.db / (np.sqrt(b_divisor) + self.epsilon)
        self.w_divisors[trainable_layer.name] = w_divisor
        self.b_divisors[trainable_layer.name] = b_divisor

class Adam(Optimizer):
    # needs checking; no improvement in loss
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.w_momentums = {}
        self.b_momentums = {}
        self.w_changes = {}
        self.b_changes = {}
        self.epsilon = epsilon
    
    def __call__(self, current_epoch, trainable_layer):
        if self.w_momentums.get(trainable_layer.name, None) == None:
            self.w_momentums[trainable_layer.name] = np.zeros_like(trainable_layer.w)
            self.b_momentums[trainable_layer.name] = np.zeros_like(trainable_layer.b)
            self.w_changes[trainable_layer.name] = np.zeros_like(trainable_layer.w)
            self.b_changes[trainable_layer.name] = np.zeros_like(trainable_layer.b)

        self.w_momentums[trainable_layer.name] = self.beta1 * self.w_momentums[trainable_layer.name] + (1 - self.beta1) * trainable_layer.dw
        self.b_momentums[trainable_layer.name] = self.beta1 * self.b_momentums[trainable_layer.name] + (1 - self.beta1) * trainable_layer.db

        w_mom_corrected = self.w_momentums[trainable_layer.name] / (1 - self.beta1 ** current_epoch)
        b_mom_corrected = self.b_momentums[trainable_layer.name] / (1 - self.beta1 ** current_epoch)

        self.w_changes[trainable_layer.name] = self.beta2 * self.w_changes[trainable_layer.name] + (1 - self.beta2) * trainable_layer.dw**2
        self.b_changes[trainable_layer.name] = self.beta2 * self.b_changes[trainable_layer.name] + (1 - self.beta2) * trainable_layer.db**2

        w_change_corrected = self.w_changes[trainable_layer.name] / (1 - self.beta2 ** current_epoch)
        b_change_corrected = self.b_changes[trainable_layer.name] / (1 - self.beta2 ** current_epoch)

        trainable_layer.w -= self.lr * w_mom_corrected / (np.sqrt(w_change_corrected) + self.epsilon)
        trainable_layer.b -= self.lr * b_mom_corrected / (np.sqrt(b_change_corrected) + self.epsilon)

class Model():
    def __init__(self, layers):
        '''
        inputs: 
            layers: list: list of Layer objects
        '''
        self.history = {'epochs':[], 'loss': [], 'last_layer_gradients': [], 'train_accuracy': [], 'validation_accuracy': []}
        if all([isinstance(layer, Layer) for layer in layers]):
            self.layers = layers # initially a list, but after __init__ converted to dict
        else:
            raise TypeError(f'not all layers are recognized. layer numbers and types: {[(i, type(l)) for i,l in enumerate(layers)]}')

    def _assign_name(self, assign_to, new_name, nametype):
        if nametype == 'name':
            assign_to.name = new_name
        elif nametype == 'prev_l':
            assign_to.prev_l = new_name
        elif nametype == 'next_l':
            assign_to.next_l = new_name

    def compile(self, n_features):
        # give each layer a unique name and also save each layer's ancestor's and descendant's name in each layer
        any([self._assign_name(self.layers[i], self.layers[i].l_type + '_' + str(i), 'name') for i in range(len(self.layers))])
        any([self._assign_name(self.layers[i], self.layers[i-1].name, 'prev_l') for i in range(1,len(self.layers))])
        any([self._assign_name(self.layers[i], self.layers[i+1].name, 'next_l') for i in range(len(self.layers)-1)])
        if len(self.layers) == 1 and isinstance(self.layers[0], Dense):
            self.layers[0].init_weights(n_features, init_type='xavier')
            # self.layers[0].w = np.random.randn(n_features, self.layers[0].n) / np.sqrt(1/n_features)
            # self.layers[0].b = np.zeros((1, self.layers[0].n)) 
        elif len(self.layers) == 1 and isinstance(self.layers[0], Conv2D):
            self.layers[0].init_weights(n_features)
            # self.layers[0].w = np.random.randn(self.layers[0].fh, self.layers[0].fw, n_features, self.layers[0].filter_num) / np.sqrt(n_features/2)
            # self.layers[0].b = np.zeros((1, self.layers[0].filter_num)) 
        for i in range(0, len(self.layers)-1):
            if isinstance(self.layers[i], Dense) and self.layers[i].trainable:
                # cur_dense_layer = self.layers[i]
                # next_layer = self.layers[i+1] # this could lead to bugs if i-1 is not an activation
                if i > 0:
                    prev_layer = self.layers[i-2]
                    if isinstance(self.layers[i+1], Relu):
                        # he initialisation if next layer is relu
                        self.layers[i].init_weights(prev_layer.n, init_type='he')
                    else:
                        # using xavier for all other activations
                        self.layers[i].init_weights(prev_layer.n, init_type='xavier')
                else:
                    if isinstance(self.layers[i+1], Relu):
                        self.layers[i].init_weights(prev_layer.n, init_type='he')
                    else:
                        self.layers[i].init_weights(prev_layer.n, init_type='xavier')
            elif isinstance(self.layers[i], Conv2D) and self.layers[i].trainable:
                self.layers[i].init_weights(prev_layer.n)
        self.layers = {self.l.name: self.l for self.l in self.layers}

    def summary(self):
        # gives overview over model in tensorflow like manner
        max_name_len = max([len(layer.name) for layer in list(self.layers.values())]) + 4 # 4 for space on either side plus two times |
        max_shape_len = max([len(layer.w.shape) for layer in list(self.layers.values()) if isinstance(layer, Dense)] + [len('(None, None)')]) + 3 # 3 for space on either side plus once | 
        string = ''
        total_params = []
        for i, layer in enumerate(self.layers.values()):
            len_name = len(layer.name)
            len_shape = len(str(layer.w.shape)) if isinstance(layer, Dense) else len('(None, None)') # amend
            if i == 0:
                string += '-' * (max_name_len + max_shape_len) 
            layer_shape = layer.w.shape if isinstance(layer, Dense) or isinstance(layer, Conv2D) else (None, None)
            string += f'\n| {layer.name} ' + ' ' * (max_name_len - len_name - 4) + '|'
            string += f' {layer_shape} ' + ' ' * (max_shape_len - len_shape - 3) + '|'
            string += '\n' + '-' * (max_name_len + max_shape_len)
            params = 0
            if layer.num_trainable_params and layer.trainable:
                params = layer.num_trainable_params
                total_params.append(params)
        string += f'\nTotal trainable parameters: {np.prod(total_params)}\n'
        print(string)

    def _generate_batches(self, x, y, batch_size):
        # generates batches of n-dimensional arrays that are shuffled
        # note that the first dimension of the array MUST be the batches (not RBG channels etc.)
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

    def _printout(self, x, i, j, epochs, batch_size, metrics, validation, time_elapsed, verbose, force_plot=False):
        if verbose < 4:
            steps = batch_size if batch_size != 1 else len(x)
            percent = (j/math.ceil(len(x)/steps))
            line = '=' * int(round((10 * percent), 0)) + ' ' * int(round((10 * (1-percent)), 0))
            string2 = f'\rstep: {j}/{math.ceil(len(x)/steps)} [{line}]; time for this epoch: {time_elapsed}\n'
            metricstring = f'\rLoss: ' + str(self.history['loss'][-1])
            if metrics:
                metricstring += '; train accuracy: ' + str(self.history['train_accuracy'][-1])
                if validation:
                    metricstring += '; validation accuracy: ' + str(self.history['validation_accuracy'][-1])
            metricstring += '\n'
        if verbose == 1:
            return metricstring
        if verbose == 2:
            sys.stdout.write(string2)
            sys.stdout.write(metricstring)
        elif verbose == 3:
            sys.stdout.write(string2)
            sys.stdout.write(metricstring)
            if i%20==0 or force_plot:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.suptitle(f'current epoch {i}')
                ax1.plot(self.history['epochs'], self.history['loss'])
                ax2.plot(self.history['epochs'], self.history['train_accuracy'], label='train')
                ax2.plot(self.history['epochs'], self.history['validation_accuracy'], label='valid')
                ax2.legend()
                ax1.set_title('loss')
                ax2.set_title('accuracy')
                ax1.set(xlabel='epochs', ylabel='loss')
                ax2.set(xlabel='epochs', ylabel='accuracy')
        else:
            if i%50 == 0:
                sys.stdout.write(f'\rEpoch: {i}/{epochs}')

    def _make_history(self, x, y, i, metrics, validation):
        self.history['epochs'].append(i)
        self.history['loss'].append(self.loss.out)
        self.history['last_layer_gradients'].append(self.loss.dout)
        if metrics:
            last_forward = list(self.layers.values())[-1]
            metric_out = metrics(last_forward.out, y, last_forward.l_type)
            self.history['train_accuracy'].append(metric_out)
            if validation:
                x_valid, y_valid = validation
                prediction_result = self.predict(x_valid)
                metric_out = metrics(prediction_result, y_valid, last_forward.l_type)
                self.history['validation_accuracy'].append(metric_out)

    def predict(self, x):
        layerlist = list(self.layers.values())
        result = x
        for layer in layerlist:
            result = layer.predict(result)
        return result

    def fit(self, x, y, epochs, loss=None, batch_size=None, validation=None, callbacks=None,
            optimizer=SGD(), metrics=Accuracy(), verbose=2):
        '''
        inputs:
            x: training examples. these should be in the format samples (rows) x features (columns)
            y: ground truth
            epochs: int: number of training epochs
            loss: Loss object
            batch_size: int (default: None): for batch training, if None it uses the full training data
            validation: tuple(np.array, np.array): tuple of array with data and ground truth for validation
            callbacks: tuple(str, float) or str: see callbacks class for further description
            optimizer: Optimizer object
            metrics: Metric object
            verbose: int: ranges 0 to 3 for increasing levels of elaboration
                          (if higher it will default to minimum printout)
        '''
        if isinstance(loss, Loss):  self.loss = loss 
        else:  raise TypeError('loss is not recognized')
        if isinstance(optimizer, Optimizer):  self.optimizer = optimizer
        else:  raise TypeError('optimizer not recognized')
        time_elapsed = 0
        layerlist = list(self.layers.values())
        calling = Callback(callbacks) if callbacks else None
        for i in range(epochs):
            start = time()
            if batch_size == None or batch_size == 1:
                batch_size = 1
                batches = self._generate_batches(x, y, len(x))
            else:
                batches = self._generate_batches(x, y, batch_size)
            # some print statements
            string = f'Epoch: {i}/{epochs}' 
            if verbose == 0:
                if i%10 == 0:
                    print(string)
            elif verbose == 1:
                if i%10 == 0 and i > 0:
                    print(string)
                    print(self._printout(x, i, 1, epochs, batch_size, metrics, validation, time_elapsed, verbose))
            elif verbose == 2:
                print(string)
            elif verbose == 3:
                print(string)
            # batches
            for j, (x_batch, y_batch) in enumerate(batches):
                # forward
                layerlist[0].forward(x_batch)
                any(layerlist[i].forward(layerlist[i-1].out) for i in range(1,len(layerlist)))
                loss.forward(layerlist[-1].out, y_batch)
                # backward
                # TODO: if layerlist[-1] softmax and loss categorical crossentropy, then use joint backwards
                loss.backward(layerlist[-1].out, y_batch) 
                layerlist[-1].backward(loss.dout) 
                if layerlist[-1].trainable:
                    self.optimizer(current_epoch=i+1, trainable_layer=layerlist[-1])
                for l in reversed(layerlist[:-1]):
                    layerlist[l].backward(layerlist[l+1].dout)
                    if layerlist[l].trainable:
                        self.optimizer(current_epoch=i+1, trainable_layer=layerlist[l])

                self._make_history(x_batch, y_batch, i+1, metrics, validation)
                self._printout(x, i+1, j+1, epochs, batch_size, metrics, validation, time_elapsed, verbose)
            # timing
            end = time()
            time_elapsed = end - start 
            # callbacks
            if calling: 
                breaking = calling(self.history)
                if breaking: 
                    last_train_acc = self.history['train_accuracy'][-1] if metrics else None
                    last_valid_acc = self.history['validation_accuracy'][-1] if metrics else None
                    print(f'\n\nCallback condition fulfilled on epoch {i}: {callbacks}; train accuracy: {last_train_acc}; validation accuracy: {last_valid_acc}')
                    self._printout(x, i+1, j+1, epochs, batch_size, metrics, validation, time_elapsed, verbose, force_plot=True)
                    break

