"""
Functions to create initializers for parameter variables
"""

from numbers import Number

import numpy as np

from .utils import floatX


class Initializer(object):
    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()


class Normal(Initializer):
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return floatX(np.random.normal(self.mean, self.std, size=shape))


class Uniform(Initializer):
    def __init__(self, range=0.01, std=None, mean=0.0):
        import warnings
        warnings.warn("The uniform initializer no longer uses Glorot et al.'s "
                      "approach to determine the bounds, but defaults to the "
                      "range (-0.01, 0.01) instead. Please use the new "
                      "GlorotUniform initializer to get the old behavior. "
                      "GlorotUniform is now the default for all layers.")

        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
            self.range = (a, b)
        elif isinstance(range, Number):
            self.range = (-range, range)
        else:
            self.range = range

    def sample(self, shape):
        return floatX(np.random.uniform(
            low=self.range[0], high=self.range[1], size=shape))


class Glorot(Initializer):
    def __init__(self, initializer=Normal, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain

    def sample(self, shape):
        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])
        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(std=std).sample(shape)


class GlorotNormal(Glorot):
    def __init__(self, gain=1.0):
        super(GlorotNormal, self).__init__(Normal, gain)


class GlorotUniform(Glorot):
    def __init__(self, gain=1.0):
        super(GlorotUniform, self).__init__(Uniform, gain)


class Glorot_c01b(Glorot):
    def sample(self, shape):
        if len(shape) != 4:
            raise RuntimeError(
                "This initializer only works with shapes of length 4")

        n1, n2 = shape[0], shape[3]
        receptive_field_size = shape[1] * shape[2]
        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(std=std).sample(shape)


class GlorotNormal_c01b(Glorot_c01b):
    def __init__(self, gain=1.0):
        super(GlorotNormal_c01b, self).__init__(Normal, gain)


class GlorotUniform_c01b(Glorot_c01b):
    def __init__(self, gain=1.0):
        super(GlorotUniform_c01b, self).__init__(Uniform, gain)


class Constant(Initializer):
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return floatX(np.ones(shape) * self.val)


class Sparse(Initializer):
    def __init__(self, sparsity=0.1, std=0.01):
        self.sparsity = sparsity
        self.std = std

    def sample(self, shape):
        if len(shape) != 2:
            raise RuntimeError(
                "sparse initializer only works with shapes of length 2")

        w = floatX(np.zeros(shape))
        n_inputs, n_outputs = shape
        size = int(self.sparsity * n_inputs)  # fraction of number of inputs

        for k in range(n_outputs):
            indices = np.arange(n_inputs)
            np.random.shuffle(indices)
            indices = indices[:size]
            values = floatX(np.random.normal(0.0, self.std, size=size))
            w[indices, k] = values

        return w


class Orthogonal(Initializer):
    """
    Orthogonal matrix initialization. For n-dimensional shapes where n > 2,
    the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.
    """
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

    def sample(self, shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return floatX(self.gain * q)
