"""
Masked versions of array API compatible arrays
"""

__version__ = "0.0.8"

import collections
import dataclasses
import importlib
import inspect
import sys
import textwrap
import types
import math

from ._mask_text import _mask_repr, _mask_str


__all__ = ["masked_namespace"]


def __getattr__(name):
    try:
        xp = importlib.import_module(name)
        mod = masked_namespace(xp)
        sys.modules[f"marray.{name}"] = mod
        return mod
    except ModuleNotFoundError as e:
        raise AttributeError(str(e))


def masked_namespace(xp):
    """Returns a masked array namespace for an Array API Standard compatible backend

    Examples
    --------
    >>> import array_api_compat.numpy as xp
    >>> from marray import masked_namespace
    >>> mxp = masked_namespace(xp)
    >>> A = mxp.asarray(xp.eye(3), mask=xp.asarray([True, False, False])[:, xp.newaxis])
    >>> x = mxp.asarray([1, 2, 3], mask=[False, False, True])
    >>> A @ x
    MArray(array([ _, 2., 0.]), array([ True, False, False]))

    """
    class MArray:

        def __init__(self, data, mask=None):
            data = xp.asarray(_get_data(data))
            mask = (xp.zeros(data.shape, dtype=xp.bool) if mask is None
                    else xp.asarray(mask, dtype=xp.bool))
            if mask.shape != data.shape:  # avoid copy if possible
                mask = xp.asarray(xp.broadcast_to(mask, data.shape), copy=True)
            self._data = data
            self._dtype = data.dtype
            self._device = getattr(data, "device", None)  # accommodate Dask
            # assert data.device == mask.device
            self._ndim = data.ndim
            self._shape = data.shape
            # accommodate PyTorch, Dask
            _size = math.prod((math.nan if i is None else i) for i in data.shape)
            self._size = None if math.isnan(_size) else _size

            self._mask = mask
            self._xp = xp

        __array_priority__ = 1  # make reflected operators work with NumPy

        @property
        def data(self):
            return self._data

        @property
        def dtype(self):
            return self._dtype

        @property
        def device(self):
            return self._device

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

        def __array_namespace__(self, api_version=None):
            if api_version is None or api_version == '2024.12':
                return mod
            else:
                message = (f"MArray interface for Array API version '{api_version}' "
                           "is not implemented.")
                raise NotImplementedError(message)

        def _call_super_method(self, method_name, *args, **kwargs):
            method = getattr(self.data, method_name)
            args = [_get_data(arg) for arg in args]
            return method(*args, **kwargs)

        def _validate_key(self, key):
            if isinstance(key, tuple):
                return tuple(self._validate_key(key_i) for key_i in key)

            if hasattr(key, 'mask') and xp.any(key.mask):
                message = ("Correct behavior for indexing with a masked array is "
                           "ambiguous, and no convention is supported at this time.")
                raise NotImplementedError(message)
            elif hasattr(key, 'mask'):
                key = key.data
            return key

        ## Indexing ##
        def __getitem__(self, key):
            key = self._validate_key(key)
            return MArray(self.data[key], self.mask[key])

        def __setitem__(self, key, other):
            key = self._validate_key(key)
            self.mask[key] = getattr(other, 'mask', False)
            return self.data.__setitem__(key, _get_data(other))

        def __iter__(self):
            return iter(self.data)

        ## Visualization ##
        def __repr__(self):
            data_str = repr(self.data)
            data_str = _mask_repr(data_str, self.mask)
            mask_str = repr(self.mask)

            if len(data_str) + len(mask_str) <= 66:
                return f"MArray({data_str}, {mask_str})"
            else:
                data_str = textwrap.indent(data_str, "    ")
                mask_str = textwrap.indent(mask_str, "    ")
                return f"MArray(\n{data_str},\n{mask_str}\n)"

        def __str__(self):
            data_str = repr(self.data)
            data_str = _mask_str(data_str, self.mask)
            return data_str

        ## Linear Algebra Methods ##
        def __matmul__(self, other):
            return mod.matmul(self, other)

        def __imatmul__(self, other):
            res = mod.matmul(self, other)
            self.data[...] = res.data[...]
            self.mask[...] = res.mask[...]
            return

        def __rmatmul__(self, other):
            other = MArray(other)
            return mod.matmul(other, self)

        ## Attributes ##

        @property
        def T(self):
            return MArray(self.data.T, self.mask.T)

        @property
        def mT(self):
            # accommodate CuPy, which lacks mT. See if it works in CuPy 14.
            has_mT = hasattr(self.data, 'mT')
            data_mT = self.data.mT if has_mT else xp.swapaxes(self.data, -1, -2)
            mask_mT = self.mask.mT if has_mT else xp.swapaxes(self.mask, -1, -2)
            return MArray(data_mT, mask_mT)
            # return MArray(self.data.mT, self.mask.mT)

        # dlpack
        def __dlpack_device__(self):
            return self.data.__dlpack_device__()

        def __dlpack__(self):
            # really not sure how to define this
            return self.data.__dlpack__()

        def to_device(self, device, /, *, stream=None):
            self._data = self._data.to_device(device, stream=stream)
            self._mask = self._mask.to_device(device, stream=stream)


    ## Methods ##

    # Methods that return the result of a unary operation as an array
    unary_names = (['__abs__', '__floordiv__', '__invert__', '__neg__', '__pos__']
                   + ['__ceil__'])
    for name in unary_names:
        def fun(self, name=name):
            data = self._call_super_method(name)
            return MArray(data, self.mask)
        setattr(MArray, name, fun)

    # Methods that return the result of a unary operation as a Python scalar
    unary_names_py = ['__bool__', '__complex__', '__float__', '__index__', '__int__']
    for name in unary_names_py:
        def fun(self, name=name):
            return self._call_super_method(name)
        setattr(MArray, name, fun)

    # Methods that return the result of an elementwise binary operation
    binary_names = ['__add__', '__sub__', '__and__', '__eq__', '__ge__', '__gt__',
                    '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
                    '__or__', '__pow__', '__rshift__', '__sub__', '__truediv__',
                    '__xor__'] + ['__divmod__', '__floordiv__']
    # Methods that return the result of an elementwise binary operation (reflected)
    rbinary_names = ['__radd__', '__rand__', '__rdivmod__', '__rfloordiv__',
                     '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__rpow__',
                     '__rrshift__', '__rsub__', '__rtruediv__', '__rxor__']
    for name in binary_names + rbinary_names:
        def fun(self, other, name=name):
            mask = (self.mask | other.mask) if hasattr(other, 'mask') else self.mask
            data = self._call_super_method(name, other)
            return MArray(data, mask)
        setattr(MArray, name, fun)

    # In-place methods
    desired_names = ['__iadd__', '__iand__', '__ifloordiv__', '__ilshift__',
                     '__imod__', '__imul__', '__ior__', '__ipow__', '__irshift__',
                     '__isub__', '__itruediv__', '__ixor__']
    for name in desired_names:
        def fun(self, other, name=name, **kwargs):
            if hasattr(other, 'mask'):
                # self.mask |= other.mask doesn't work because mask has no setter
                self.mask.__ior__(other.mask)
            self._call_super_method(name, other)
            return self
        setattr(MArray, name, fun)

    mod_name = f'marray.{xp.__name__}'
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod

    mod.MArray = MArray

    ## Constants ##
    constant_names = ['e', 'inf', 'nan', 'newaxis', 'pi']
    for name in constant_names:
        setattr(mod, name, getattr(xp, name))

    ## Creation Functions ##
    def asarray(obj, /, *, mask=None, dtype=None, device=None, copy=None):
        if device is not None:
            raise NotImplementedError("`device` argument is not implemented")

        data = _get_data(obj)
        data = xp.asarray(data, dtype=dtype, device=device, copy=copy)

        mask = (getattr(obj, 'mask', xp.full(data.shape, False))
                if mask is None else mask)
        mask = xp.asarray(mask, dtype=xp.bool, device=device, copy=copy)

        return MArray(data, mask=mask)
    mod.asarray = asarray

    creation_functions = ['arange', 'empty', 'eye', 'from_dlpack',
                          'full', 'linspace', 'ones', 'zeros']
    creation_functions_like = ['empty_like', 'full_like', 'ones_like', 'zeros_like']
    #  handled with array manipulation functions
    creation_manip_functions = ['tril', 'triu', 'meshgrid']
    for name in creation_functions:
        def fun(*args, name=name, **kwargs):
            data = getattr(xp, name)(*args, **kwargs)
            return MArray(data)
        setattr(mod, name, fun)

    for name in creation_functions_like:
        def fun(x, /, *args, name=name, **kwargs):
            data = getattr(xp, name)(_get_data(x), *args, **kwargs)
            return MArray(data, mask=getattr(x, 'mask', False))
        setattr(mod, name, fun)

    ## Data Type Functions and Data Types ##
    dtype_fun_names = ['can_cast', 'finfo', 'iinfo', 'result_type']
    for name in dtype_fun_names:
        def fun(*args, name=name, **kwargs):
            args = [_get_data(arg) for arg in args]
            return getattr(xp, name)(*args, **kwargs)
        setattr(mod, name, fun)

    dtype_names = ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                   'uint32', 'uint64', 'float32', 'float64', 'complex64', 'complex128',
                   'isdtype']  # not really a dtype, but OK to treat it like one here
    inspection_fun_names = ['__array_namespace_info__']
    version_attribute_names = ['__array_api_version__']
    for name in (dtype_names + inspection_fun_names + version_attribute_names):
        setattr(mod, name, getattr(xp, name))

    def astype(x, dtype, /, *, copy=True, device=None):
        if device is None and not copy and dtype == x.dtype:
            return x
        x = asarray(x)
        data = xp.astype(x.data, dtype, copy=copy, device=device)
        mask = xp.astype(x.mask, xp.bool, copy=copy, device=device)
        return MArray(data, mask=mask)
    mod.astype = astype

    ## Elementwise Functions ##
    elementwise_names = ['abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan',
                         'atan2', 'atanh', 'bitwise_and', 'bitwise_left_shift',
                         'bitwise_invert', 'bitwise_or', 'bitwise_right_shift',
                         'bitwise_xor', 'ceil', 'conj', 'copysign', 'cos',
                         'cosh', 'divide', 'equal', 'exp', 'expm1', 'floor',
                         'floor_divide', 'greater', 'greater_equal', 'hypot',
                         'imag', 'isfinite', 'isinf', 'isnan', 'less', 'less_equal',
                         'log', 'log1p', 'log2', 'log10', 'logaddexp', 'logical_and',
                         'logical_not', 'logical_or', 'logical_xor', 'maximum',
                         'minimum', 'multiply', 'negative', 'nextafter', 'not_equal',
                         'positive', 'pow', 'real', 'reciprocal', 'remainder', 'round',
                         'sign', 'signbit', 'sin', 'sinh', 'square', 'sqrt', 'subtract',
                         'tan', 'tanh', 'trunc']
    for name in elementwise_names:
        def fun(*args, name=name, **kwargs):
            masks = [arg.mask for arg in args if hasattr(arg, 'mask')]
            masks = xp.broadcast_arrays(*masks)
            args = [_get_data(arg) for arg in args]
            out = getattr(xp, name)(*args, **kwargs)
            return MArray(out, mask=xp.any(xp.stack(masks), axis=0))
        setattr(mod, name, fun)


    def clip(x, /, min=None, max=None):
        args = [x, min, max]
        masks = [arg.mask for arg in args if hasattr(arg, 'mask')]
        masks = xp.broadcast_arrays(*masks)
        mask = xp.any(xp.stack(masks), axis=0)
        datas = [_get_data(arg) for arg in args]
        data = xp.clip(datas[0], min=datas[1], max=datas[2])
        return MArray(data, mask)
    mod.clip = clip

    ## Indexing Functions
    def take(x, indices, /, *, axis=None):
        indices_data = _get_data(indices)
        shape = xp.asarray(indices_data).shape
        indices_mask = getattr(indices, 'mask', xp.broadcast_to(xp.asarray(False), shape))
        indices_data[indices_mask] = 0  # ensure valid index
        data = xp.take(x.data, indices_data, axis=axis)
        mask = xp.take(x.mask, indices_data, axis=axis) | indices_mask
        return MArray(data, mask=mask)
    mod.take = take

    def take_along_axis(x, indices, /, *, axis=-1):
        indices_data = _get_data(indices)
        indices_mask = getattr(indices, 'mask', False)
        indices_data[indices_mask] = 0  # ensure valid index
        data = xp.take_along_axis(x.data, indices_data, axis=axis)
        mask = xp.take_along_axis(x.mask, indices_data, axis=axis) | indices_mask
        return MArray(data, mask=mask)
    mod.take_along_axis = take_along_axis

    ## Inspection ##
    # Included with dtype functions above

    ## Linear Algebra Functions ##
    def get_linalg_fun(name):
        def linalg_fun(x1, x2, /, **kwargs):
            x1 = asarray(x1)
            x2 = asarray(x2)
            zero = xp.asarray(0)
            data1 = xp.where(x1.mask, xp.asarray(0, dtype=x1.dtype), x1.data)
            data2 = xp.where(x2.mask, xp.asarray(0, dtype=x2.dtype), x2.data)
            fun = getattr(xp, name)
            data = fun(data1, data2, **kwargs)
            # Strict array can't do arithmetic with booleans
            # mask = ~fun(~x1.mask, ~x2.mask)
            mask = fun(xp.astype(~x1.mask, xp.float32),
                       xp.astype(~x2.mask, xp.float32), **kwargs)
            mask = ~xp.astype(mask, xp.bool)
            return MArray(data, mask)
        return linalg_fun

    linalg_names = ['matmul', 'tensordot', 'vecdot']
    for name in linalg_names:
        setattr(mod, name, get_linalg_fun(name))

    mod.matrix_transpose = lambda x: x.mT

    ## Manipulation Functions ##
    first_arg_arrays = {'broadcast_arrays', 'concat', 'stack', 'meshgrid'}
    output_arrays = {'broadcast_arrays', 'unstack', 'meshgrid'}

    def get_manip_fun(name):
        def manip_fun(x, *args, **kwargs):
            x = (asarray(x) if name not in first_arg_arrays
                 else [asarray(xi) for xi in x])
            mask = (x.mask if name not in first_arg_arrays
                    else [xi.mask for xi in x])
            data = (x.data if name not in first_arg_arrays
                    else [xi.data for xi in x])

            fun = getattr(xp, name)

            if name == "repeat":
                args = list(args)
                repeats = args[0]
                if hasattr(repeats, 'mask') and xp.any(repeats.mask):
                    message = ("Correct behavior when `repeats` is a masked array is "
                               "ambiguous, and no convention is supported at this time.")
                    raise NotImplementedError(message)
                elif hasattr(repeats, 'mask'):
                    repeats = repeats.data
                args[0] = repeats

            if name in {'broadcast_arrays', 'meshgrid'}:
                res = fun(*data, *args, **kwargs)
                mask = fun(*mask, *args, **kwargs)
            else:
                res = fun(data, *args, **kwargs)
                mask = fun(mask, *args, **kwargs)

            out = (MArray(res, mask) if name not in output_arrays
                   else tuple(MArray(resi, maski) for resi, maski in zip(res, mask)))
            return out
        return manip_fun

    manip_names = ['broadcast_arrays', 'broadcast_to', 'concat', 'expand_dims',
                   'flip', 'moveaxis', 'permute_dims', 'repeat', 'reshape',
                   'roll', 'squeeze', 'stack', 'tile', 'unstack']
    for name in manip_names + creation_manip_functions:
        setattr(mod, name, get_manip_fun(name))
    mod.broadcast_arrays = lambda *arrays: get_manip_fun('broadcast_arrays')(arrays)
    mod.meshgrid = lambda *arrays, **kwargs: get_manip_fun('meshgrid')(arrays, **kwargs)

    ## Searching Functions
    def searchsorted(x1, x2, /, *, side='left', sorter=None):
        if sorter is not None:
            x1 = take(x1, sorter)

        mask_count = xp.cumulative_sum(xp.astype(x1.mask, xp.int64))
        x1_compressed = x1.data[~x1.mask]
        count = xp.zeros(x1_compressed.shape[0]+1, dtype=xp.int64)
        count[:-1] = mask_count[~x1.mask]
        count[-1] = count[-2]
        i = xp.searchsorted(x1_compressed, x2.data, side=side)
        j = i + xp.take(count, i)
        return MArray(j, mask=x2.mask)

    def nonzero(x, /):
        x = asarray(x)
        data = xp.asarray(x.data, copy=True)
        data[x.mask] = 0
        res = xp.nonzero(data)
        return tuple(MArray(resi) for resi in res)

    def where(condition, x1, x2, /):
        condition = asarray(condition)
        data1 = _get_data(x1)  # do not prematurely convert Python scalar to array
        data2 = _get_data(x2)
        x1 = asarray(x1)
        x2 = asarray(x2)
        data = xp.where(condition.data, data1, data2)
        mask = xp.where(condition.data,
                        condition.mask | x1.mask,
                        condition.mask | x2.mask)
        return MArray(data, mask)

    mod.searchsorted = searchsorted
    mod.nonzero = nonzero
    mod.where = where

    # Defined below, in Statistical Functions
    # argmax
    # argmin
    # count_nonzero

    ## Set Functions ##
    def get_set_fun(name):
        def set_fun(x, /):
            sentinel = _xinfo(x).max
            any_masked = xp.any(x.mask)
            if any_masked and xp.any((x.data == sentinel) & ~x.mask):
                message = (f"The maximum value of the data's dtype is included in the "
                           "non-masked data; this complicates the isolation of unique "
                           "non-masked values when masked values are present. Consider "
                           "promoting to another dtype to use `{name}`.")
                raise NotImplementedError(message)
            x = asarray(x)
            data = xp.asarray(x.data, copy=True)
            # Replace masked elements with a sentinel value: they are all treated as
            # the same as one another and distinct from all non-masked values.
            data[x.mask] = xp.asarray(sentinel, dtype=data.dtype)
            fun = getattr(xp, name)
            res = fun(data)
            if name == 'unique_values':
                return MArray(res, res == sentinel) if any_masked else MArray(res)

            fields = res._fields
            name_tuple = res.__class__.__name__
            result_class = collections.namedtuple(name_tuple, fields)

            mask = (res.values == sentinel) if any_masked else None
            result_list = []
            for res_i, field_i in zip(res, fields):
                mask_i = None if field_i in {'inverse_indices', 'indices'} else mask
                result_list.append(MArray(res_i, mask=mask_i))
            return result_class(*result_list)

        return set_fun

    unique_names = ['unique_values', 'unique_counts', 'unique_inverse', 'unique_all']
    for name in unique_names:
        setattr(mod, name, get_set_fun(name))

    ## Sorting Functions ##
    def get_sort_fun(name):
        def sort_fun(x, /, *, axis=-1, descending=False, stable=True):
            x = asarray(x)
            data = xp.asarray(x.data, copy=True)
            sentinel = _xinfo(x).min if descending else _xinfo(x).max
            any_masked = xp.any(x.mask)
            if any_masked and xp.any((data == sentinel) & ~x.mask):
                minmax = "minimum" if descending else "maximum"
                message = (f"The {minmax} value of the data's dtype is included in the "
                           "non-masked data; this complicates sorting when masked values "
                           "are present. Consider promoting to another dtype to use "
                           f"`{name}`.")
                raise NotImplementedError(message)
            data[x.mask] = xp.asarray(sentinel, dtype=data.dtype)
            fun = getattr(xp, name)
            kwargs = {'descending': True} if descending else {}
            res = fun(data, axis=axis, stable=stable, **kwargs)
            mask = (res == sentinel) if (name=='sort') and any_masked else None
            return MArray(res, mask)
        return sort_fun

    sort_names = ['sort', 'argsort']
    for name in sort_names:
        setattr(mod, name, get_sort_fun(name))

    ## Statistical Functions and Utility Functions ##
    def get_statistical_fun(name):
        def statistical_fun(x, *args, axis=None, name=name, **kwargs):
            replacements = {'max': _xinfo(x).min,
                            'min': _xinfo(x).max,
                            'sum': 0,
                            'count_nonzero': 0,
                            'prod': 1,
                            'argmax': _xinfo(x).min,
                            'argmin': _xinfo(x).max,
                            'all': True,
                            'any': False}
            x = asarray(x)
            data = xp.asarray(x.data, copy=True)
            data[x.mask] = replacements[name]
            fun = getattr(xp, name)
            res = fun(data, *args, axis=axis, **kwargs)
            mask = xp.all(x.mask, axis=axis, keepdims=kwargs.get('keepdims', False))
            return MArray(res, mask=mask)
        return statistical_fun

    def count(x, axis=None, keepdims=False):
        x = asarray(x)
        not_mask = xp.astype(~x.mask, xp.uint64)
        return xp.sum(not_mask, axis=axis, keepdims=keepdims, dtype=xp.uint64)

    def _cumulative_op(x, *args, _identity, _op, **kwargs):
        x = asarray(x)
        axis = kwargs.get('axis', None)
        if axis is None:
            x = mod.reshape(x, -1)

        data = xp.asarray(x.data, copy=True)
        mask = x.mask
        data[mask] = _identity
        res = _op(data, *args, **kwargs)

        if kwargs.get('include_initial', False):
            # get `false` of the correct dimensionality to prepend
            shape = xp.take(res, xp.asarray([0]), axis=axis).shape
            false = xp.full(shape, False, dtype=xp.bool)
            mask = xp.concat((false, mask), axis=axis)

        return MArray(res, mask)

    def cumulative_sum(x, *args, **kwargs):
        return _cumulative_op(x, *args, _identity=0, _op=xp.cumulative_sum, **kwargs)

    def cumulative_prod(x, *args, **kwargs):
        return _cumulative_op(x, *args, _identity=1, _op=xp.cumulative_prod, **kwargs)

    def mean(x, axis=None, keepdims=False):
        s = mod.sum(x, axis=axis, keepdims=keepdims)
        dtype = xp.uint64 if xp.isdtype(s.dtype, ("bool", "integral")) else s.dtype
        n = mod.astype(mod.count(x, axis=axis, keepdims=keepdims), dtype)
        return s / n

    def var(x, axis=None, correction=0, keepdims=False):
        m = mod.mean(x, axis=axis, keepdims=True)
        xm = x - m
        xmc = mod.conj(xm) if mod.isdtype(xm.dtype, 'complex floating') else xm
        s = mod.sum(xm*xmc, axis=axis, keepdims=keepdims)
        dtype = xp.uint64 if xp.isdtype(s.dtype, ("bool", "integral")) else s.dtype
        n = mod.astype(mod.count(x, axis=axis, keepdims=keepdims), dtype)
        out = s / (n - correction)
        out = mod.real(out) if mod.isdtype(xm.dtype, 'complex floating') else out
        return out

    def diff(x, /, axis=-1, n=1, prepend=None, append=None):
        # Array API specifies that prepend and append must be arrays,
        # the shapes must be the same as `x` except for the length along `axis`,
        # and the dtypes must match. Satisfy the API now; optionally,
        # respect underlying backend's preferences regarding type promotion
        # scalar support, etc., later.
        x = [asarray(el) for el in (prepend, x, append) if el is not None]

        x = mod.concat(x, axis=axis) if len(x) > 1 else x[0]
        x = mod.moveaxis(x, axis, -1) if axis != -1 else x

        data = xp.diff(x.data)
        mask = x.mask
        if mask.shape[-1] > 1:
            mask = mask[..., :-1] | mask[..., 1:]
        out = asarray(data, mask=mask)

        # "Higher-order differences must be calculated recursively..."
        if n > 1:
            out = mod.diff(out, axis=-1, n=n-1)

        return mod.moveaxis(out, -1, axis) if axis != -1 else out

    mod.count = count
    mod.mean = mean
    mod.var = var
    mod.std = lambda *args, **kwargs: mod.var(*args, **kwargs)**0.5
    mod.diff = diff

    search_names = ['argmax', 'argmin', 'count_nonzero']
    statfun_names = ['max', 'min', 'sum', 'prod']
    utility_names = ['all', 'any']
    for name in search_names + statfun_names + utility_names:
        setattr(mod, name, get_statistical_fun(name))
    mod.cumulative_sum = cumulative_sum
    mod.cumulative_prod = cumulative_prod

    preface = ["The following is the documentation for the corresponding "
               f"attribute of `{xp.__name__}`.",
               "The behavior of `marray` is the same when the mask is all "
               "`False`, and differs",
               "(as described in the tutorial) when the mask has `True` elements.\n\n"]
    preface = "\n".join(preface)
    for attribute in mod.__dict__.keys():
        # Add documentation if it is not already present
        if getattr(mod, attribute).__doc__:
            continue

        xp_attr = getattr(xp, attribute, None)
        mod_attr = getattr(mod, attribute, None)
        if xp_attr is not None and mod_attr is not None:

            if hasattr(xp_attr, "__doc__"):
                try:
                    xp_doc = getattr(xp_attr, "__doc__")
                    getattr(mod, attribute).__doc__ = preface + xp_doc
                except (AttributeError, TypeError):
                    pass

            try:
                mod_attr.__signature__ = inspect.signature(xp_attr)
            except (ValueError, TypeError):
                pass

            try:
                mod_attr.__name__ = xp_attr.__name__
            except (AttributeError, TypeError):
                pass

    return mod


_get_namespace = masked_namespace


def _xinfo(x):
    xp = x._xp
    if xp.isdtype(x.dtype, 'integral'):
        return xp.iinfo(x.dtype)
    elif xp.isdtype(x.dtype, 'bool'):
        binfo = dataclasses.make_dataclass("binfo", ['min', 'max'])
        return binfo(min=False, max=True)
    else:
        return xp.finfo(x.dtype)


def _get_data(x):
    # get data from an MArray or NumPy masked array *without*
    # getting memoryview from NumPy array, etc.
    return x.data if hasattr(x, 'mask') else x
