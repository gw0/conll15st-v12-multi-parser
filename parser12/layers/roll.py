# https://github.com/fchollet/keras/pull/607/files

from keras import backend as K
import theano.tensor as T  #XXX
from keras.layers.core import Layer, MaskedLayer
from keras import activations, initializations, regularizers, constraints


class Roll(Layer):
    '''
    Convenience function to roll `TensorType`s along the given axis.

    Syntax copies numpy.roll function.

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    shift : int (symbolic or literal)
        The number of places by which elements are shifted.
    axis : int (symbolic or literal), optional
        The axis along which elements are shifted. By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    tensor
        Output tensor, with the same shape as `x`.
        Mimics numpy.roll
    '''
    def __init__(self, shift, axis=None):
        super(Roll, self).__init__()
        self.shift = shift
        self.axis = axis

    def get_output(self, train):
        shift = self.shift
        axis = self.axis

        x = self.get_input(train)
        return T.roll(x, shift, axis=axis)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "shift": self.shift,
                "axis": self.axis}


class RollOffsets(MaskedLayer):
    '''
    Convenience function to roll `TensorType`s along the given axis to multiple offsets.

    Like `numpy.roll` function with multiple offsets support.

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    offsets : tuple of ints (symbolic or literal)
        Tuple of numbers of places by which elements are shifted.
    axis : int (symbolic or literal)
        The axis along which elements are rolled/shifted.
    offset_axis : int (symbolic or literal)
        The axis where different offsets will be placed.

    Returns
    -------
    tensor
        Output tensor, similar shape as `x`, but with an additional offset dimension after given axis.
    '''
    def __init__(self, offsets, axis=None, offset_axis=None):
        if axis is None:
            axis = 1
        if offset_axis is None:
            offset_axis = axis + 1
        super(RollOffsets, self).__init__()

        self.axis = axis
        self.offsets = offsets
        self.offset_axis = offset_axis

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return input_shape[0:self.offset_axis] + (len(self.offsets),) + input_shape[self.offset_axis:]

    def get_output(self, train):
        X = self.get_input(train)
        tensors = [ T.roll(X, off, axis=self.axis)  for off in self.offsets ]
        return T.stack(tensors, axis=self.offset_axis)

    def get_output_mask(self, train=False):
        X = self.get_input_mask(train)
        if X is None:
            return None
        tensors = [ T.roll(X, off, axis=self.axis)  for off in self.offsets ]
        return T.stack(tensors, axis=self.offset_axis)

    def get_config(self):
        config = {
            "name": self.__class__.__name__,
            "axis": self.axis,
            "offsets": self.offsets,
            "offset_axis": self.offset_axis,
        }
        base_config = super(RollOffsets, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RepeatVector2(MaskedLayer):
    '''
        Repeat input n times.
        Dimensions of input are assumed to be (nb_samples, dim).
        Return tensor of shape (nb_samples, n, dim).
    '''
    def __init__(self, n, axis=1, **kwargs):
        super(RepeatVector2, self).__init__(**kwargs)
        self.n = n
        self.axis = axis

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return input_shape[0:self.axis] + (self.n,) + input_shape[self.axis:]

    def get_output(self, train=False):
        X = self.get_input(train)
        tensors = [X] * self.n
        return T.stack(tensors, axis=self.axis)

    def get_output_mask(self, train=False):
        M = self.get_input_mask(train)
        tensors = [M] * self.n
        return T.stack(tensors, axis=self.axis)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "n": self.n}
        base_config = super(RepeatVector2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedMerge2(Layer):
    '''Sum/multiply/average over the outputs of a TimeDistributed layer.

    mode: {'sum', 'mul', 'ave'}
    Tensor input dimensions:   (nb_sample, time, features)
    Tensor output dimensions:  (nb_sample, features)
    '''

    def __init__(self, mode='sum', axis=1, **kwargs):
        super(TimeDistributedMerge2, self).__init__(**kwargs)
        self.mode = mode
        self.axis = axis
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

    @property
    def output_shape(self):
        return (self.input_shape[:self.axis], self.input_shape[self.axis + 1:])

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.mode == 'ave':
            s = K.mean(X, axis=self.axis)
            return s
        if self.mode == 'sum':
            s = K.sum(X, axis=self.axis)
            return s
        elif self.mode == 'mul':
            s = K.mul(X, axis=self.axis)
            return s
        else:
            raise Exception('Unknown merge mode')

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "mode": self.mode,
                  "axis": self.axis}
        base_config = super(TimeDistributedMerge2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedDense2(MaskedLayer):
    '''
       Apply a same Dense layer for each dimension[1] (time_dimension) input.
       Especially useful after a recurrent network with 'return_sequence=True'.
       Tensor input dimensions:   (nb_sample, time_dimension, input_dim)
       Tensor output dimensions:  (nb_sample, time_dimension, output_dim)
    '''

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(TimeDistributedDense2, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[-1]

        self.W = self.init((input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))

        self.params = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return input_shape[:-1] + (self.output_dim,)

    def get_output(self, train=False):
        X = self.get_input(train)
        axis = len(self.input_shape) - 1
        permutation = range(axis - 1, -1, -1) + [axis]
        output = self.activation(K.dot(X.dimshuffle(permutation), self.W) + self.b)
        return output.dimshuffle(permutation)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(TimeDistributedDense2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransparentMaskMerge(MaskedLayer):
    """
    Overlay layers and merge them by treating masks as transparent.

    Arguments:

        - layers: input layers or containers with same shape (like in Merge), or None when using Graph model with `merge_mode='join'`
    """

    def __init__(self, layers=None, **kwargs):
        super(TransparentMaskMerge, self).__init__(**kwargs)
        self.layers = layers

    def get_output_with_mask(self, train=False):
        if self.layers is None:
            # from Graph model with `merge_mode='join'`
            inputs = self.get_input(train).values()
            #inputs_mask = self.get_input_mask(train).values()
            inputs_mask = [ layer.get_output_mask(train) for layer in self.previous.layers ]  #XXX: workaround
        else:
            # from Sequential model with `layers` argument
            inputs = [ layer.get_output(train) for layer in self.layers ]
            inputs_mask = [ layer.get_output_mask(train) for layer in self.layers ]

        X = inputs[0]
        M = inputs_mask[0]
        X = X * M  # reapply mask just in case
        for X2, M2 in zip(inputs[1:], inputs_mask[1:]):
            X += X2 * M2 * (True ^ M)
            M |= M2
        return X, M

    def get_output(self, train=False):
        return self.get_output_with_mask(train)[0]

    def get_output_mask(self, train=False):
        return self.get_output_with_mask(train)[1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "layers": self.layers}
        base_config = super(TransparentMaskMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

