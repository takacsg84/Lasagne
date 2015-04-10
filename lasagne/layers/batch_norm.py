# -*- coding: utf-8 -*-

"""
Preliminary implementation of batch normalization for Lasagne.
Does not include a way to properly compute the normalization factors over the
full training set for testing, but can be used as a drop-in for training and
validation.

Author: Jan SchlÃ¼ter
"""

import numpy as np
import theano
import theano.tensor as T
from .. import nonlinearities
from .base import Layer


class BatchNormLayer(Layer):

    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.05,
                 nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).

        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        self.inference_updates = None
        if nonlinearity is None:
            nonlinearity = nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.count = theano.shared(np.dtype(theano.config.floatX).type(0),
                                   'count')
        self.mean = theano.shared(np.zeros(shape, dtype=dtype), 'mean')
        self.std = theano.shared(np.ones(shape, dtype=dtype), 'std')
        self.beta = theano.shared(np.zeros(shape, dtype=dtype), 'beta')
        self.gamma = theano.shared(np.ones(shape, dtype=dtype), 'gamma')

    def get_params(self):
        return [self.gamma] + self.get_bias_params()

    def get_bias_params(self):
        return [self.beta]

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std

        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)

    def additional_updates(self, input=None, **kwargs):
        kwargs['deterministic'] = True
        input = self.input_layer.get_output(input, **kwargs)
        mean = input.mean(self.axes, keepdims=True)
        std = input.std(self.axes, keepdims=True)

        new_count = self.count + 1
        new_mean = (self.mean * self.count + mean) / new_count
        new_std = (self.std * self.count + std) / new_count
        return[(self.mean, new_mean),
               (self.std, new_std),
               (self.count, new_count)]

    def reset(self):
        dtype = theano.config.floatX
        shape = self.mean.get_value().shape
        self.mean = theano.shared(np.zeros(shape, dtype=dtype), 'mean')
        self.std = theano.shared(np.ones(shape, dtype=dtype), 'std')


def batch_norm(layer):
    """
    Convenience function to apply batch normalization to a given layer's
    output. Will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), and will
    remove the layer's bias if there is one (because it would be redundant).

    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b'):
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity)
