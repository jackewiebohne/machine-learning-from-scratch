def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)
    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))
    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = None
        self.bias = None
        self.cache = None

    #     self._init_weights()

    # def _init_weights(self):
    #     self.weight = np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size) #* 1e-3 
    #     print(self.weight.shape)
    #     # self.bias = np.zeros(self.out_channels)
    #     # self.weight = np.random.randn(2, 2, 3, 8) # (f, f, n_C_prev, n_C)
    #     self.bias = np.random.randn(self.out_channels)

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        windows = getWindows(x, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride)
        out = np.einsum('bihwkl,oikl->bohw', windows, self.weight)

        # add bias to kernels
        out += self.bias[None, :, None, None]
        self.cache = x, windows
        return out

    def backward(self, dout):
        x, windows = self.cache
        padding = self.kernel_size - 1 if self.padding == 0 else self.padding
        dout_windows = getWindows(dout, x.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride - 1)
        rot_kern = np.rot90(self.weight, 2, axes=(2, 3))

        db = np.sum(dout, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', windows, dout)
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        return db, dw, dx

#################
def zero_pad(X, pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant', constant_values = (0,0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W)
    Z = np.sum(s)
    b = np.squeeze(b)
    Z = Z + b
    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H = int((n_H_prev + 2*pad - f)/stride) + 1
    n_W = int((n_W_prev + 2*pad - f)/stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]          # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            vert_start = stride * h 
            vert_end = vert_start  + f            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                horiz_start = stride * w
                horiz_end = horiz_start + f
                for c in range(n_C):   # loop over channels (= #filters) of the output volume
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    weights = W[:, :, :, c]
                    biases  = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros(A_prev.shape)                          
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape) 
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    vert_start = stride * h 
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    return dA_prev, dW, db

##########
def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  i0 = np.repeat(np.arange(field_height,dtype='int32'), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height,dtype='int32'), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width,dtype='int32'), int(out_height))
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C,dtype='int32'), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]



class Conv():

    def __init__(self, X_dim, n_filter, h_filter, w_filter, stride, padding):

        self.d_X, self.h_X, self.w_X = X_dim

        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter
        self.stride, self.padding = stride, padding

        self.W = None # np.random.randn(n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.)
        self.b = None # np.random.randn(self.n_filter, 1) 
        self.params = [self.W, self.b]

        self.h_out = (self.h_X - h_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_X - w_filter + 2 * padding) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.n_filter, self.h_out, self.w_out)

    def forward(self, X):

        self.n_X = X.shape[0]

        self.X_col = im2col_indices(
            X, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        W_row = self.W.reshape(self.n_filter, -1)

        out = W_row @ self.X_col + self.b
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        W_flat = self.W.reshape(self.n_filter, -1)

        dX_col = W_flat.T @ dout_flat
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = col2im_indices(dX_col, shape, self.h_filter,
                            self.w_filter, self.padding, self.stride)

        return dX, dW, db
######
class own_Conv2D():
    l_type = 'conv2D'
    def __init__(self, filter_num, filter_sizes=(3,3), stride=1, pad_size=2, pad_val=0):
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
            self.stride = stride
            self.pad_val = pad_val
            self.prev_l = None
            self.next_l = None
            self.inputs = None
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

    def _pad(self, array, pad_size, pad_val):
        '''
        only symmetric padding is possible
        ''' 
        return np.pad(array, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', constant_values=(pad_val, pad_val))


    def _dilate(self, array, stride_size, pad_size, symmetric_filter_shape, output_image_size):
        # on dilation for backprop with stride>1, 
        # see: https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
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
        expects inputs to be of shape: [batchsize, height, width, channel in]
        after init, filter_shapes are: [fh, fw, channel in, channel out] 
        '''
        self.input_shape = x.shape
        x_pad = self._pad(x, self.pad_size, self.pad_val)
        self.input_pad_shape = x_pad.shape
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
        # for dW we'll use the column approach with ordinary dot product for variety ;) tensordot does the same without all the reshaping
        dout_descendant_flat = dout_descendant.reshape(-1, self.Cout) # new shape (batch_size * Hout * Wout, Cout)
        x_flat = self.inputs.reshape(-1, self.fh * self.fw * Cin) # shape (batch_size * Hout * Wout, fh * fw * Cin)
        dw = x_flat.T @ dout_descendant_flat # shape (fh * fw * Cin, Cout)
        self.dw = dw.reshape(self.fh, self.fw, Cin, self.Cout)
        del dout_descendant_flat # free memory
        # for dinputs: we'll get padded and dilated windows of dout_descendant and perform the tensordot with 180 rotated W
        # for details, see https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710 ; also: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c ; also: https://youtu.be/Lakz2MoHy6o?t=835
        Wrot180 = np.rot90(self.w, 2, axes=(0,1)) # or also self.w[::-1, ::-1, :, :]
        # backprop for forward with stride > 1 is done on windowed dout that's padded and dilated with stride 1
        dout_descendant = self._dilate(dout_descendant, stride_size=self.stride, pad_size=self.pad_size, symmetric_filter_shape=self.fh, output_image_size=h)
        dout_descendant = self._pad(dout_descendant, pad_size=self.fw-1, pad_val=self.pad_val) # pad dout_descendant to dim: fh-1 (or fw-1); only symmetrical filters are supported
        dout_descendant = self._windows(array=dout_descendant, stride_size=1, filter_shapes=(self.fh, self.fw),
                            out_height=h + 2 * self.pad_size, out_width=w + 2 * self.pad_size) # shape: (batch_size * h_padded * w_padded, fh * fw * Cout)
        self.dout = np.tensordot(dout_descendant, Wrot180, axes=([3,4,5],[0,1,3]))
        self.dout = self.dout[:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size, :]
        ## einsum alternative, but slower:
        # dinput = np.einsum('nhwfvk,fvck->nhwc', dout_windows, self.W)

#####
# class Convolution2D():
#     # strides are not implemented
#     def __init__(self, weights=None, bias=None, shape=None, *kwargs):

#         if weights is None:
#             assert shape is not None, 'Both weights and shape cannot be None'
#             self.filter_size, _, self.num_channels, self.num_filters = shape
#             weights = np_random_normal(0, 1 / np.sqrt(self.filter_size * self.filter_size * self.num_channels),
#                                        size=(self.filter_size,
#                                              self.filter_size,
#                                              self.num_channels,
#                                              self.num_filters) )
#             self.bias = np.zeros((1,1,1, self.num_channels))

#         self.filter_size, _, self.num_channels, self.num_filters = weights.shape
#         self.weights = weights
#         self.bias = bias
#         self.filter_size, _, self.num_channels, self.num_filters = self.weights.shape

#     def feed_forward(self, X_batch, **kwargs):
#         # if self.first_feed_forward:  # First run
#         self.first_feed_forward = False
#         self.batch_size = len(X_batch)
#         self.image_size = X_batch.shape[1]

#         # The following is used as argument to out of ufuncs

#         self.input_conv = np.zeros((self.batch_size, self.image_size,
#                                     self.image_size,
#                                     self.num_filters))

#         # creating the zero padding structure once is efficient

#         self.image_size_embedding_size = self.image_size + self.filter_size - 1
#         self.input_zero_padded = np.zeros((self.batch_size,
#                                            self.image_size_embedding_size,
#                                            self.image_size_embedding_size,
#                                            self.num_channels))

#         z = np.arange(0, self.image_size)
#         zs = np.stack([z + i for i in range(self.weights.shape[0])], 1)
#         self.batch_index = np.arange(self.batch_size)[:, None, None, None, None, None, None]
#         self.channel_index = np.arange(self.num_channels)[None, None, None, None, None, :, None]
#         self.filter_index = np.arange(self.num_filters)[None, None, None, None, None, None, :]
#         self.rows = zs[None, :, None, :, None, None, None]
#         self.cols = zs[None, None, :, None, :, None, None]
#         self.tmp = np.zeros(shape=(self.batch_size,
#                                    self.image_size,
#                                    self.image_size,
#                                    self.filter_size,
#                                    self.filter_size,
#                                    self.num_channels,
#                                    self.num_filters) )

#         self.input = X_batch

#         # Convolution
#         print(self.filter_size // 2, self.filter_size // 2 + 1)
#         print(self.image_size_embedding_size)
#         print(self.input_zero_padded.shape)
#         self.input_zero_padded[:,
#         self.filter_size // 2:-self.filter_size // 2 + 1,
#         self.filter_size // 2:-self.filter_size // 2 + 1] = self.input

#         # TODO: better to loose the last index from all the fancy indices
#         # and do a tensordot
#         # Also compare with reshape and a np.matmul
#         np.multiply(self.input_zero_padded[self.batch_index,
#                                            self.rows,
#                                            self.cols,
#                                            self.channel_index],
#                     self.weights[None, None, None, :, :, :, :],
#                     out=self.tmp)
#         self.tmp.sum(axis=(3, 4, 5), out=self.input_conv)

#         self.output = self.input_conv + self.bias
#         return self.output

#     def back_prop(self, loss_d_output):
#         if self.first_back_prop:
#             self.first_back_prop = False

#             # The following three are used with out parameter of ufuncs
#             self.loss_d_output_times_output_d = np.zeros_like(self.output)
#             self.loss_derivative_input = np.zeros_like(self.input)
#             self.loss_derivative_weights = np.zeros_like(self.weights)
#             self.loss_d_output_times_output_d_zero_padded = np.zeros((self.batch_size,
#                                                                       self.image_size_embedding_size,
#                                                                       self.image_size_embedding_size,
#                                                                       self.num_filters))

#         # np.multiply(loss_d_output, self.output_d,
#         #            out=self.loss_d_output_times_output_d)
#         self.loss_d_output_times_output_d = loss_d_output

#         # correction for weights
#         if self.trainable:

#             self.input_zero_padded[:,
#             self.filter_size // 2:-self.filter_size // 2 + 1,
#             self.filter_size // 2:-self.filter_size // 2 + 1] = self.input

#             (self.input_zero_padded[self.batch_index, self.rows, self.cols, self.channel_index] *
#              self.loss_d_output_times_output_d[:, :, :, None, None, None, :]).sum(axis=(0, 1, 2), out=self.loss_derivative_weights)

#         # if not self.first_layer:
#         self.loss_d_output_times_output_d_zero_padded[:,
#             self.filter_size // 2:-self.filter_size // 2 + 1,
#             self.filter_size // 2:-self.filter_size // 2 + 1] = self.loss_d_output_times_output_d

#         np.multiply(self.loss_d_output_times_output_d_zero_padded[self.batch_index,
#                                                                   self.rows,
#                                                                   self.cols,
#                                                                   self.filter_index],
#                     self.weights[None, None, None, ::-1, ::-1, :, :],
#                     out=self.tmp
#                     )
#         self.tmp.sum(axis=(3, 4, 6), out=self.loss_derivative_input)
#         self.loss_derivative_bias = np.sum(loss_d_output, axis=(0,1,2), keepdims=True)
#         return self.loss_derivative_input, self.loss_derivative_weights, self.loss_derivative_bias

#####
import numpy as np
from scipy import signal
from time import time, time_ns
np.random.seed(1)

# hyper params
batch_size = 100000
in_h = 4
in_w = 4
in_channels = 3
out_channels = 8
kernel_size = 2
stride = 3
padding = 2
timer = {'im2col':[], 'slow reference':0, 'myversion': []}

# set weigths and biases
W = np.random.randn(kernel_size, kernel_size, in_channels, out_channels).astype(np.float32) # (f, f, n_C_prev, n_C)
b = np.random.randn(1, 1, 1, out_channels).astype(np.float32)

# set inputs
A_prev = np.random.randn(batch_size, in_h, in_w, in_channels).astype(np.float32)
x = A_prev.transpose(0,3,1,2)

# # init params for slow reference conv
# hparameters = {"pad" : padding,
#                "stride": stride}
# s = time()
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# dA, dweights, dbias = conv_backward(Z, cache_conv)
# e = time()
# timer['slow reference'] = e-s

# # init ordinary conv
# conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding) # in_channels, out_channels, kernel_size=3, stride=1, padding=0
# conv.weight = W.transpose(3,2,0,1) # self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size
# conv.bias = np.squeeze(b[-1])
# conv_out = conv.forward(x)
# db, dw, dx = conv.backward(conv_out)

for _ in range(10):
# init im2col conv
    conv2 = Conv(X_dim=(in_channels, in_h, in_w), n_filter=out_channels, h_filter=kernel_size, w_filter=kernel_size, stride=stride, padding=padding)
    conv2.W = W.transpose(3,2,0,1) 
    conv2.b = np.expand_dims(np.squeeze(b[-1]),1)
    ims = time()
    z = conv2.forward(x)
    dX, dW, dB = conv2.backward(z)
    ime = time()
    timer['im2col'].append(ime-ims)

    # init my own conv
    conv3 = own_Conv2D(out_channels, filter_sizes=(kernel_size, kernel_size), stride=stride, pad_size=padding) # n, filter_sizes=(3,3), stride=1, pad_size=2, pad_val=0
    conv3.w = W
    conv3.b = b
    ms = time()
    conv3.forward(A_prev)
    conv3.backward(conv3.out)
    dinputs = conv3.dout
    dmyweights = conv3.dw
    dmybias = conv3.db
    me = time()
    timer['myversion'].append(me-ms)
print('my version', np.mean(timer['myversion']), 'im2col', np.mean(timer['im2col']))

# init chowdhury's convs
# if stride ==1:
#     conv4 = Convolution2D(weights=W, bias=b)
#     fw = conv4.feed_forward(A_prev)
#     derloss, derw, derb = conv4.back_prop(loss_d_output)


# print test result
# print('slvrfn conv version dx, dw, db', np.mean(dx), np.mean(dw), np.mean(db)) # dx mean mostly larger than reference with stride 2; everything else identical to reference
# print('my own version  dinputs, dmyweights, dmybias', np.mean(dinputs), np.mean(dmyweights), np.mean(dmybias)) # dinpunts mostly smaller than reference with stride 2; everything else identical to reference
# print('im2col version dX, dW, dB',np.mean(dX), np.mean(dW), np.mean(dB)) # identical to reference
# if stride == 1: print('chowdhury version dX, dw, db', np.mean(derloss), np.mean(derw), np.mean(derb))
# print("reference dA, dweights, dbias", np.mean(dA), np.mean(dweights), np.mean(dbias))