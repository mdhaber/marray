"""
Masked versions of array API compatible arrays
"""

__version__ = "0.0.4"

import numpy as np  # temporarily used in __repr__ and __str__
from . import formatting

def masked_array(xp):
    """Returns a masked array namespace for an array API backend

    Examples
    --------
    >>> from scipy._lib.array_api_compat import numpy as xp
    >>> from scipy.stats import masked_array
    >>> ma = masked_array(xp)
    >>> A = ma.eye(3)
    >>> A.mask[0, ...] = True
    >>> x = ma.asarray([1, 2, 3], mask=[False, False, True])
    >>> A @ x
    masked_array(data=[--, 2.0, 0.0],
                 mask=[ True, False, False],
           fill_value=1e+20)

    """
    class MaskedArray():

        def __init__(self, data, mask=None):
            data = getattr(data, '_data', data)
            mask = (xp.zeros(data.shape, dtype=bool) if mask is None
                    else xp.asarray(mask, dtype=bool))
            mask = xp.asarray(xp.broadcast_to(mask, data.shape), copy=True)
            self._data = data
            self._dtype = data.dtype
            self._ndim = data.ndim
            self._shape = data.shape
            self._size = data.size

            self._mask = mask
            self._xp = xp
            self._sentinel = (info(self).max if not xp.isdtype(self.dtype, 'bool')
                              else None)

        @property
        def data(self):
            return self._data

        @property
        def dtype(self):
            return self._dtype

        @property
        def ndim(self):
            return self._ndim

        @property
        def shape(self):
            return self._shape

        @property
        def size(self):
            return self._size

        @property
        def mask(self):
            return self._mask

        def call_super_method(self, method_name, *args, **kwargs):
            method = getattr(self.data, method_name)
            args = [getattr(arg, 'data', arg) for arg in args]
            return method(*args, **kwargs)

        ## Indexing ##
        def __getitem__(self, key):
            return MaskedArray(self.data[key], self.mask[key])

        def __setitem__(self, key, other):
            self.mask[key] = getattr(other, 'mask', False)
            return self.data.__setitem__(key, getattr(other, 'data', other))

        ## Visualization ##
        def __repr__(self):
            return formatting.format_repr(self)

        def __str__(self):
            return formatting.format_data(self.data, self.mask)

        ## Linear Algebra Methods ##
        def __matmul__(self, other):
            return mod.matmul(self, other)

        def __rmatmul__(self, other):
            return mod.matmul(self, other)

        ## Attributes ##

        @property
        def T(self):
            return MaskedArray(self.data.T, self.mask.T)

        @property
        def mT(self):
            return MaskedArray(self.data.mT, self.mask.mT)

    ## Methods ##

    # Methods that return the result of a unary operation as an array
    unary_names = (['__abs__', '__floordiv__', '__invert__', '__neg__', '__pos__']
                   + ['__ceil__'])
    for name in unary_names:
        def fun(self, name=name):
            data = self.call_super_method(name)
            return MaskedArray(data, self.mask)
        setattr(MaskedArray, name, fun)

    # Methods that return the result of a unary operation as a Python scalar
    unary_names_py = ['__bool__', '__complex__', '__float__', '__index__', '__int__']
    for name in unary_names_py:
        def fun(self, name=name):
            return self.call_super_method(name)
        setattr(MaskedArray, name, fun)

    # Methods that return the result of an elementwise binary operation
    binary_names = ['__add__', '__sub__', '__and__', '__eq__', '__ge__', '__gt__',
                    '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
                    '__or__', '__pow__', '__rshift__', '__sub__', '__truediv__',
                    '__xor__'] + ['__divmod__', '__floordiv__']
    # Methods that return the result of an elementwise binary operation (reflected)
    rbinary_names = ['__radd__', '__rand__', '__rdivmod__', '__rfloordiv__',
                     '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__rpow__',
                     '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__',
                     '__rxor__']
    for name in binary_names + rbinary_names:
        def fun(self, other, name=name):
            mask = (self.mask | other.mask) if hasattr(other, 'mask') else self.mask
            data = self.call_super_method(name, other)
            return MaskedArray(data, mask)
        setattr(MaskedArray, name, fun)

    # In-place methods
    desired_names = ['__iadd__', '__iand__', '__ifloordiv__', '__ilshift__',
                     '__imod__', '__imul__', '__ior__', '__ipow__', '__irshift__',
                     '__isub__', '__itruediv__', '__ixor__']
    for name in desired_names:
        def fun(self, other, name=name, **kwargs):
            if hasattr(other, 'mask'):
                # self.mask |= other.mask doesn't work because mask has no setter
                self.mask.__ior__(other.mask)
            self.call_super_method(name, other)
            return self
        setattr(MaskedArray, name, fun)

    # Inherited

    # To be added
    # __array_namespace__
    # __dlpack__, __dlpack_device__
    # to_device?

    def info(x):
        xp = x._xp
        if xp.isdtype(x.dtype, 'integral'):
            return xp.iinfo(x.dtype)
        else:
            return xp.finfo(x.dtype)

    class module:
        pass

    mod = module()

    ## Constants ##
    constant_names = ['e', 'inf', 'nan', 'newaxis', 'pi']
    for name in constant_names:
        setattr(mod, name, getattr(xp, name))

    ## Creation Functions ##
    def asarray(obj, /, *, mask=None, dtype=None, device=None, copy=None):
        if device is not None:
            raise NotImplementedError()

        data = getattr(obj, 'data', obj)
        mask = (getattr(obj, 'mask', xp.full(data.shape, False))
                if mask is None else mask)

        data = xp.asarray(data, dtype=dtype, device=device, copy=copy)
        mask = xp.asarray(mask, dtype=dtype, device=device, copy=copy)
        return MaskedArray(data, mask=mask)
    mod.asarray = asarray

    creation_functions = ['arange', 'empty', 'empty_like', 'eye', 'from_dlpack',
                          'full', 'full_like', 'linspace', 'meshgrid', 'ones',
                          'ones_like', 'tril', 'triu', 'zeros', 'zeros_like']
    for name in creation_functions:
        def fun(*args, name=name, **kwargs):
            data = getattr(xp, name)(*args, **kwargs)
            return MaskedArray(data)
        setattr(mod, name, fun)

    ## Data Type Functions and Data Types ##
    dtype_fun_names = ['can_cast', 'finfo', 'iinfo', 'isdtype', 'result_type']
    dtype_names = ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                   'uint32', 'uint64', 'float32', 'float64', 'complex64', 'complex128']
    for name in dtype_fun_names + dtype_names:
        setattr(mod, name, getattr(xp, name))

    mod.astype = (lambda x, dtype, /, *, copy=True, **kwargs:
                  asarray(x, copy=copy or (dtype != x.dtype), dtype=dtype, **kwargs))

    ## Elementwise Functions ##
    elementwise_names = ['abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan',
                         'atan2', 'atanh', 'bitwise_and', 'bitwise_left_shift',
                         'bitwise_invert', 'bitwise_or', 'bitwise_right_shift',
                         'bitwise_xor', 'ceil', 'clip', 'conj', 'copysign', 'cos',
                         'cosh', 'divide', 'equal', 'exp', 'expm1', 'floor',
                         'floor_divide', 'greater', 'greater_equal', 'hypot',
                         'imag', 'isfinite', 'isinf', 'isnan', 'less', 'less_equal',
                         'log', 'log1p', 'log2', 'log10', 'logaddexp', 'logical_and',
                         'logical_not', 'logical_or', 'logical_xor', 'maximum',
                         'minimum', 'multiply', 'negative', 'not_equal', 'positive',
                         'pow', 'real', 'remainder', 'round', 'sign', 'signbit',
                         'sin', 'sinh', 'square', 'sqrt', 'subtract', 'tan', 'tanh',
                         'trunc']
    for name in elementwise_names:
        def fun(*args, name=name, **kwargs):
            masks = [arg.mask for arg in args if hasattr(arg, 'mask')]
            masks = xp.broadcast_arrays(*masks)
            args = [getattr(arg, 'data', arg) for arg in args]
            out = getattr(xp, name)(*args, **kwargs)
            return MaskedArray(out, mask=xp.any(masks, axis=0))
        setattr(mod, name, fun)

    ## Indexing Functions
    # To be written:
    # take

    def xp_take_along_axis(arr, indices, axis):
        # This is just for regular arrays; not masked arrays
        arr = xp_swapaxes(arr, axis, -1)
        indices = xp_swapaxes(indices, axis, -1)

        m = arr.shape[-1]
        n = indices.shape[-1]

        shape = list(arr.shape)
        shape.pop(-1)
        shape = shape + [n,]

        arr = xp.reshape(arr, (-1,))
        indices = xp.reshape(indices, (-1, n))

        offset = (xp.arange(indices.shape[0]) * m)[:, xp.newaxis]
        indices = xp.reshape(offset + indices, (-1,))

        out = arr[indices]
        out = xp.reshape(out, shape)
        return xp_swapaxes(out, axis, -1)
    mod._xp_take_along_axis = xp_take_along_axis

    ## Inspection ##
    # To be written
    # __array_namespace_info
    # capabilities
    # default_device
    # default_dtypes
    # devices
    # dtypes

    ## Linear Algebra Functions ##
    def get_linalg_fun(name):
        def linalg_fun(x1, x2, /, **kwargs):
            x1 = asarray(x1)
            x2 = asarray(x2)
            data1 = xp.asarray(x1.data, copy=True)
            data2 = xp.asarray(x2.data, copy=True)
            data1[x1.mask] = 0
            data2[x2.mask] = 0
            fun = getattr(xp, name)
            data = fun(data1, data2)
            mask = ~fun(~x1.mask, ~x2.mask)
            return MaskedArray(data, mask)
        return linalg_fun

    linalg_names = ['matmul', 'tensordot', 'vecdot']
    for name in linalg_names:
        setattr(mod, name, get_linalg_fun(name))

    ## Manipulation Functions ##
    first_arg_arrays = {'broadcast_arrays', 'concat', 'stack'}
    output_arrays = {'broadcast_arrays', 'unstack'}

    def get_manip_fun(name):
        def manip_fun(x, *args, **kwargs):
            x = (asarray(x) if name not in first_arg_arrays
                 else [asarray(xi) for xi in x])
            mask = (x.mask if name not in first_arg_arrays
                    else [xi.mask for xi in x])
            data = (x.data if name not in first_arg_arrays
                    else [xi.data for xi in x])

            fun = getattr(xp, name)

            if name == 'broadcast_arrays':
                res = fun(*data, *args, **kwargs)
                mask = fun(*mask, *args, **kwargs)
            else:
                res = fun(data, *args, **kwargs)
                mask = fun(mask, *args, **kwargs)

            out = (MaskedArray(res, mask) if name not in output_arrays
                   else [MaskedArray(resi, maski) for resi, maski in zip(res, mask)])
            return out
        return manip_fun

    manip_names = ['broadcast_arrays', 'broadcast_to', 'concat', 'expand_dims',
                   'flip', 'moveaxis', 'permute_dims', 'repeat', 'reshape',
                   'roll', 'squeeze', 'stack', 'tile', 'unstack']
    for name in manip_names:
        setattr(mod, name, get_manip_fun(name))
    mod.broadcast_arrays = lambda *arrays: get_manip_fun('broadcast_arrays')(arrays)

    # This is just for regular arrays; not masked arrays
    def xp_swapaxes(arr, axis1, axis2):
        axes = list(range(arr.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return xp.permute_dims(arr, axes)
    mod.xp_swapaxes = xp_swapaxes

    ## Searching Functions
    # To be added
    # searchsorted

    def nonzero(x, /):
        x = asarray(x)
        data = xp.asarray(x.data, copy=True)
        data[x.mask] = 0
        res = xp.nonzero(data)
        return tuple(MaskedArray(resi) for resi in res)

    def where(condition, x1, x2, /):
        condition = asarray(condition)
        x1 = asarray(x1)
        x2 = asarray(x2)
        data = xp.where(condition.data, x1.data, x2.data)
        mask = condition.mask | x1.mask | x2.mask
        return MaskedArray(data, mask)

    mod.nonzero = nonzero
    mod.where = where

    # Defined below, in Statistical Functions
    # argmax
    # argmin

    ## Set Functions ##
    def get_set_fun(name):
        def set_fun(x, /):
            # This seems a little inconsistent with nonzero and where, which
            # completely ignore masked elements.
            x = asarray(x)
            data = xp.asarray(x.data, copy=True)
            data[x.mask] = x._sentinel
            fun = getattr(xp, name)
            res = fun(data)
            # this sort of works but could be refined
            return (MaskedArray(res, res==x._sentinel) if name=='unique_values'
                    else tuple(MaskedArray(resi, resi==x._sentinel) for resi in res))
        return set_fun

    unique_names = ['unique_values', 'unique_counts', 'unique_inverse', 'unique_all']
    for name in unique_names:
        setattr(mod, name, get_set_fun(name))

    ## Sorting Functions ##
    def get_sort_fun(name):
        def sort_fun(x, /, *, axis=-1, descending=False, stable=True):
            x = asarray(x)
            data = xp.asarray(x.data, copy=True)
            sentinel = info(x).min if descending else info(x).max
            data[x.mask] = sentinel
            fun = getattr(xp, name)
            res = fun(data, axis=axis, descending=descending, stable=stable)
            mask = (res == sentinel) if name=='sort' else None
            return MaskedArray(res, mask)
        return sort_fun

    sort_names = ['sort', 'argsort']
    for name in sort_names:
        setattr(mod, name, get_sort_fun(name))

    ## Statistical Functions and Utility Functions ##
    def get_statistical_fun(name):
        def statistical_fun(x, *args, axis=None, name=name, **kwargs):
            replacements = {'max': info(x).min,
                            'min': info(x).max,
                            'sum': 0,
                            'prod': 1,
                            'argmax': info(x).min,
                            'argmin': info(x).max,
                            'all': xp.asarray(True),
                            'any': xp.asarray(False)}
            x = asarray(x)
            data = xp.asarray(x.data, copy=True)
            data[x.mask] = replacements[name]
            fun = getattr(xp, name)
            res = fun(data, *args, axis=axis, **kwargs)
            mask = xp.all(x.mask, axis=axis)
            return MaskedArray(res, mask=mask)
        return statistical_fun

    def count(x, axis=None, keepdims=False):
        x = asarray(x)
        return xp.sum(x.mask, axis=axis, keepdims=keepdims)

    def cumulative_sum(x, *args, **kwargs):
        x = asarray(x)
        data = xp.asarray(x.data, copy=True)
        data[x.mask] = 0
        res = xp.cumulative_sum(data, *args, **kwargs)
        return MaskedArray(res, x.mask)

    def mean(x, axis=None, keepdims=False):
        s = mod.sum(x, axis=axis, keepdims=keepdims)
        n = mod.count(x, axis=axis, keepdims=keepdims)
        return s / n

    def var(x, axis=None, correction=0, keepdims=False):
        # rewrite this to use xp.var but replace masked entries with mean.
        m = mod.mean(x, axis=axis, keepdims=True)
        xm = x - m
        n = mod.count(x, axis=axis, keepdims=keepdims)
        s = mod.sum(xm**2, axis=axis, keepdims=keepdims)
        return s / (n - correction)

    mod.count = count
    mod.mean = mean
    mod.var = var
    mod.std = lambda *args, **kwargs: np.sqrt(mod.var(*args, **kwargs))

    search_names = ['argmax', 'argmin']
    statfun_names = ['max', 'min', 'sum', 'prod']
    utility_names = ['all', 'any']
    for name in search_names + statfun_names + utility_names:
        setattr(mod, name, get_statistical_fun(name))
    mod.cumulative_sum = cumulative_sum

    return mod
