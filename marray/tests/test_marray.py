import functools
import inspect
import operator
import re

import pytest
import marray

import numpy as np
import array_api_strict as strict
xps = [np, strict]
xps_take_along_axis = []

try:
    from array_api_compat import torch
    torch.set_default_dtype(torch.float64)
    xps.append(torch)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from array_api_compat import cupy
    xps.append(cupy)
except (ImportError, ModuleNotFoundError):
    pass

dtypes_boolean = ['bool']
dtypes_uint = ['uint8', 'uint16', 'uint32', 'uint64', ]
dtypes_int = ['int8', 'int16', 'int32', 'int64']
dtypes_real = ['float32', 'float64']
dtypes_complex = ['complex64', 'complex128']
dtypes_integral = dtypes_uint + dtypes_int
dtypes_numeric = dtypes_integral + dtypes_real + dtypes_complex
dtypes_all = dtypes_boolean + dtypes_integral + dtypes_real + dtypes_complex


def as_numpy(x):
    # Use `cupy.testing` when `assert_allclose` and `assert_equal` support `strict`
    try:
        return np.asarray(x)
    except:
        return cupy.asnumpy(x)


def pass_backend(*, xp, pass_xp, fun=None, pass_funs=None, dtype=None,
                 pass_dtypes=None, pass_fun=pytest.skip, reason="Debug later."):
    pass_funs = [None] if pass_funs is None else pass_funs
    pass_dtypes = [None] if pass_dtypes is None else pass_dtypes
    if pass_xp in str(xp) and dtype in pass_dtypes and fun in pass_funs:
        pass_fun(reason=reason)


def get_arrays(n_arrays, *, dtype, xp, shape=None, ndim=(1, 4),
               pre_broadcasted=False, seed=None):
    xpm = marray.masked_namespace(xp)

    entropy = np.random.SeedSequence(seed).entropy
    rng = np.random.default_rng(entropy)

    ndim = rng.integers(*ndim) if isinstance(ndim, tuple) else ndim
    shape = rng.integers(1, 20, size=ndim) if shape is None else np.asarray(shape)

    datas = []
    masks = []
    for i in range(n_arrays):
        shape_mask = rng.random(size=ndim) > 0.85
        shape_i = shape.copy()
        if not pre_broadcasted:
            shape_i[shape_mask] = 1
        data = rng.standard_normal(size=shape_i)

        if dtype == 'bool':
            data = data > 0
        elif str(dtype).startswith('complex'):
            data = (data * 10 + rng.standard_normal(size=shape_i) * 10j).astype(dtype)
        else:
            # multiply by 10 to get some variety in integers
            data = (data*10).astype(dtype)

        datas.append(data)
        # for now, make masks same shape as array
        # consider making them broadcastable to array shape
        # or broadcastable to the same shape
        mask = rng.random(size=shape_i) > 0.75
        masks.append(mask)

    marrays = []
    masked_arrays = []
    for array, mask in zip(datas, masks):
        marrays.append(xpm.asarray(array, mask=mask))
        masked_arrays.append(np.ma.masked_array(array, mask=mask))

    return marrays, masked_arrays, entropy


def assert_comparison(res, ref, seed, xp, comparison, **kwargs):
    array_type = type(xp.asarray(1.))
    assert isinstance(res.data, array_type)
    assert isinstance(res.mask, array_type)
    try:
        ref_mask = ref.mask.__array_namespace__().broadcast_to(ref.mask, ref.data.shape)
    except:  # CuPy doesn't have `__array_namespace__`. Try to simplify in CuPy 14.
        ref_mask = xp.broadcast_to(ref.mask, ref.data.shape)
    try:
        strict = kwargs.pop('strict', True)
        res_data = res.data[~res.mask]
        ref_data = ref.data[~ref_mask]
        comparison(as_numpy(res_data), as_numpy(ref_data), strict=strict, **kwargs)
        comparison(as_numpy(res.mask), as_numpy(ref_mask), strict=True, **kwargs)
    except AssertionError as e:
        raise AssertionError(seed) from e


def assert_equal(res, ref, seed, xp=None, **kwargs):
    return assert_comparison(res, ref, seed, xp, np.testing.assert_equal, **kwargs)


def assert_allclose(res, ref, seed, xp=None, **kwargs):
    low_precision = xp.isdtype(res.dtype, (xp.float32, xp.complex64))
    tol = {'rtol': 1e-5, 'atol': 1e-16} if low_precision else {'rtol': 1e-10, 'atol': 1e-32}
    kwargs = tol | kwargs
    return assert_comparison(res, ref, seed, xp, np.testing.assert_allclose, **kwargs)


def pass_exceptions(allowed=[]):
    def outer(f):
        @functools.wraps(f)
        def inner(*args, seed=None, **kwargs):
            try:
                return f(*args, seed=seed, **kwargs)
            except (AttributeError, ValueError, TypeError,
                    NotImplementedError, RuntimeError) as e:
                for message in allowed:
                    if re.escape(message) in re.escape(str(e)):
                        pytest.xfail(str(e))
                else:
                    raise AssertionError(seed) from e
        return inner
    return outer


def get_rtol(dtype, xp):
    dtype = getattr(xp, dtype)
    if xp.isdtype(dtype, ('real floating', 'complex floating')):
        return xp.finfo(dtype).eps**0.5
    else:
        return 0


arithmetic_unary = {"+x": lambda x: +x, "-x": lambda x: -x, "abs": abs}
arithmetic_methods_unary = {"x.__abs__": lambda x: x.__abs__(),
                            "x.__neg__": lambda x: x.__neg__(),
                            "x.__pos__": lambda x: x.__pos__()}
arithmetic_binary = {"x + y": lambda x, y: x + y,
                     "x - y": lambda x, y: x - y,
                     "x * y": lambda x, y: x * y,
                     "x / y": lambda x, y: x / y,
                     "x // y": lambda x, y: x // y,
                     "x % y": lambda x, y: x % y,
                     "x ** y": lambda x, y: x ** y}
arithmetic_methods_binary = {"x.__add__(y)": lambda x, y: x.__add__(y),
                             "x.__floordiv__(y)": lambda x, y: x.__floordiv__(y),
                             "x.__mod__(y)": lambda x, y: x.__mod__(y),
                             "x.__mul__(y)": lambda x, y: x.__mul__(y),
                             "x.__pow__(y)": lambda x, y: x.__pow__(y),
                             "x.__sub__(y)": lambda x, y: x.__sub__(y),
                             "x.__truediv__(y)": lambda x, y: x.__truediv__(y)}
array_binary = {"x @ y": lambda x, y: x @ y,
                "operator.matmul": operator.matmul,
                "opterator.__matmul__": operator.__matmul__}
array_methods_binary = {"x.__matmul__(y)": lambda x, y: x.__matmul__(y)}
bitwise_unary = {'bitwise_invert': lambda x: ~x}
bitwise_methods_unary = {'bitwise_invert': lambda x: x.__invert__()}
bitwise_binary = {'bitwise_and': lambda x, y: x & y,
                  'bitwise_or': lambda x, y: x | y,
                  'bitwise_xor': lambda x, y: x ^ y,
                  'bitwise_left_shift': lambda x, y: x << y,
                  'bitwise_right_shift': lambda x, y: x >> y}
bitwise_methods_binary = {'bitwise_and': lambda x, y: x.__and__(y),
                          'bitwise_or': lambda x, y: x.__or__(y),
                          'bitwise_left_shift': lambda x, y: x.__lshift__(y),
                          'bitwise_right_shift': lambda x, y: x.__rshift__(y),
                          'bitwise_xor': lambda x, y: x.__xor__(y)}

scalar_conversions = {bool: True, int: 10, float: 1.5, complex: 1.5 + 2.5j}

# tested in test_dlpack
# __dlpack__, __dlpack_device__, to_device
# tested in test_indexing
# __getitem__, __index__, __setitem__,

comparison_binary = {"x < y": lambda x, y: x < y,
                     "x <= y": lambda x, y: x <= y,
                     "x > y": lambda x, y: x > y,
                     "x >= y": lambda x, y: x >= y,
                     "x == y": lambda x, y: x == y ,
                     "x != y": lambda x, y: x != y}
comparison_methods_binary = {"x.__eq__(y)": lambda x, y: x.__eq__(y),
                             "x.__ge__(y)": lambda x, y: x.__ge__(y),
                             "x.__gt__(y)": lambda x, y: x.__gt__(y),
                             "x.__le__(y)": lambda x, y: x.__le__(y),
                             "x.__lt__(y)": lambda x, y: x.__lt__(y),
                             "x.__ne__(y)": lambda x, y: x.__ne__(y)}

def iadd(x, y): x += y
def isub(x, y): x -= y
def imul(x, y): x *= y
def itruediv(x, y): x /= y
def ifloordiv(x, y): x //= y
def ipow(x, y): x **= y
def imod(x, y): x %= y
inplace_arithmetic = [iadd, isub, imul, itruediv, ifloordiv, ipow, imod]

def imatmul(x, y): x @= y
inplace_array = [imatmul]

def iand(x, y): x &= y
def ior(x, y): x |= y
def ixor(x, y): x ^= y
def ilshift(x, y): x <<= y
def irshift(x, y): x >>= y
inplace_bitwise = [iand, ior, ixor, ilshift, irshift]


data_type_fun = ['can_cast', 'finfo', 'iinfo', 'result_type']
inspection = ['__array_namespace_info__']
version = ['__array_api_version__']
elementwise_unary = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
                     'ceil', 'conj', 'cos', 'cosh', 'exp', 'expm1', 'floor', 'imag',
                     'isfinite', 'isinf', 'isnan', 'log', 'log1p', 'log2', 'log10',
                     'logical_not', 'negative', 'positive', 'real', 'reciprocal',
                     'round', 'sign', 'signbit', 'sin', 'sinh', 'square', 'sqrt',
                     'tan', 'tanh', 'trunc']
elementwise_binary = ['add', 'atan2', 'copysign', 'divide', 'equal', 'floor_divide',
                      'greater', 'greater_equal', 'hypot', 'less', 'less_equal',
                      'logaddexp', 'logical_and', 'logical_or', 'logical_xor',
                      'maximum', 'minimum', 'multiply', 'nextafter', 'not_equal',
                      'pow', 'remainder', 'subtract']
searching_array = ['argmax', 'argmin', 'count_nonzero']
statistical_array = ['cumulative_sum', 'cumulative_prod', 'max', 'mean',
                     'min', 'prod', 'std', 'sum', 'var']
utility_array = ['all', 'any']


torch_exceptions = ["\"abs_cpu\" not implemented for 'Bool",
                    "\"abs_cpu\" not implemented for 'UInt",
                    "\"add_stub\" not implemented for 'UInt",
                    "\"addmm_impl_cpu_\" not implemented for 'Bool",
                    "\"arange_cpu\" not implemented for 'Bool",
                    "\"arange_cpu\" not implemented for 'UInt",
                    "\"arange_cpu\" not implemented for 'Complex",
                    "\"argmin_cpu\" not implemented for 'Bool",
                    "\"argmax_cpu\" not implemented for 'Bool",
                    "\"argmin_cpu\" not implemented for 'Complex",
                    "\"argmax_cpu\" not implemented for 'Complex",
                    "\"atan2_cpu\" not implemented for 'UInt",
                    "\"atan2_cpu\" not implemented for 'Complex",
                    "\"bitwise_and_cpu\" not implemented for 'UInt",
                    "\"bitwise_or_cpu\" not implemented for 'UInt",
                    "\"bitwise_not_cpu\" not implemented for 'UInt",
                    "\"bitwise_xor_cpu\" not implemented for 'UInt",
                    "\"bmm\" not implemented for 'Bool'",
                    "\"ceil_vml_cpu\" not implemented for 'Bool",
                    "\"copysign_cpu\" not implemented for 'Complex",
                    "\"div_floor_cpu\" not implemented for 'Bool",
                    "\"div_floor_cpu\" not implemented for 'Complex",
                    "\"div_floor_cpu\" not implemented for 'UInt",
                    "\"flip_cpu\" not implemented for 'UInt",
                    "\"floor_vml_cpu\" not implemented for 'Bool",
                    "\"ge_cpu\" not implemented for 'Complex",
                    "\"ge_cpu\" not implemented for 'UInt",
                    "\"gt_cpu\" not implemented for 'Complex",
                    "\"gt_cpu\" not implemented for 'UInt",
                    "\"hypot_cpu\" not implemented for 'Bool",
                    "\"hypot_cpu\" not implemented for 'Byte",
                    "\"hypot_cpu\" not implemented for 'Char",
                    "\"hypot_cpu\" not implemented for 'Complex",
                    "\"hypot_cpu\" not implemented for 'Int",
                    "\"hypot_cpu\" not implemented for 'Long",
                    "\"hypot_cpu\" not implemented for 'Short",
                    "\"hypot_cpu\" not implemented for 'UInt",
                    "\"index_cpu\" not implemented for 'UInt",
                    "\"le_cpu\" not implemented for 'Complex",
                    "\"le_cpu\" not implemented for 'UInt",
                    "\"linspace_cpu\" not implemented for 'Bool",
                    "\"linspace_cpu\" not implemented for 'UInt",
                    "\"logaddexp_cpu\" not implemented for 'Bool",
                    "\"logaddexp_cpu\" not implemented for 'Byte",
                    "\"logaddexp_cpu\" not implemented for 'Char",
                    "\"logaddexp_cpu\" not implemented for 'Int",
                    "\"logaddexp_cpu\" not implemented for 'Long",
                    "\"logaddexp_cpu\" not implemented for 'Short",
                    "\"logaddexp_cpu\" not implemented for 'UInt",
                    "\"logical_and_cpu\" not implemented for 'UInt",
                    "\"logical_or_cpu\" not implemented for 'UInt",
                    "\"logical_not_cpu\" not implemented for 'UInt",
                    "\"logical_xor_cpu\" not implemented for 'UInt",
                    "\"lshift_cpu\" not implemented for 'UInt",
                    "\"lshift_cpu\" not implemented for 'Bool",
                    "\"lt_cpu\" not implemented for 'Complex",
                    "\"lt_cpu\" not implemented for 'UInt",
                    "\"masked_fill\" not implemented for 'UInt",
                    "\"maximum_cpu\" not implemented for 'UInt",
                    "\"max_values_cpu\" not implemented for 'Complex",
                    "\"minimum_cpu\" not implemented for 'UInt",
                    "\"min_values_cpu\" not implemented for 'Complex",
                    "\"neg_cpu\" not implemented for 'UInt",
                    "\"nextafter_cpu\" not implemented for 'Bool",
                    "\"nextafter_cpu\" not implemented for 'Byte",
                    "\"nextafter_cpu\" not implemented for 'Char",
                    "\"nextafter_cpu\" not implemented for 'Complex",
                    "\"nextafter_cpu\" not implemented for 'Int",
                    "\"nextafter_cpu\" not implemented for 'Long",
                    "\"nextafter_cpu\" not implemented for 'Short",
                    "\"nextafter_cpu\" not implemented for 'UInt",
                    "\"remainder_cpu\" not implemented for 'Bool",
                    "\"remainder_cpu\" not implemented for 'UInt",
                    "\"remainder_cpu\" not implemented for 'Complex",
                    "\"round_vml_cpu\" not implemented for 'Bool",
                    "\"round_vml_cpu\" not implemented for 'Complex",
                    "\"rshift_cpu\" not implemented for 'UInt",
                    "\"rshift_cpu\" not implemented for 'Bool",
                    "\"sum_cpu\" not implemented for 'UInt",
                    "\"pow\" not implemented for 'Bool",
                    "\"pow\" not implemented for 'UInt",
                    "\"searchsorted_out_cpu\" not implemented for 'UInt",
                    "\"sign_cpu\" not implemented for 'UInt",
                    "\"signbit_cpu\" not implemented for 'UInt",
                    "\"signbit_cpu\" not implemented for 'Complex",
                    "\"tril\" not implemented for 'UInt",
                    "\"triu\" not implemented for 'UInt",
                    "\"trunc_vml_cpu\" not implemented for 'Bool",
                    "\"unique\" not implemented for 'Complex",
                    "\"where_cpu\" not implemented for 'UInt",
                    "ceil is not supported for complex inputs",
                    "floor is not supported for complex inputs",
                    "imag is not implemented for tensors with non-complex",
                    "maximum not implemented for complex tensors.",
                    "minimum not implemented for complex tensors.",
                    "Negation, the `-` operator, on a bool tensor is not supported",
                    "signbit is not implemented for complex tensors.",
                    "Subtraction, the `-` operator, with two bool tensors",
                    "The `+` operator, on a bool tensor is not supported",
                    "trunc is not supported for complex inputs",
                    "module 'array_api_compat.torch' has no attribute 'repeat'",
                    "torch.reshape doesn't yet support the copy keyword",
                    "unique_all() not yet implemented for pytorch",
                    ]


@pytest.mark.parametrize("f_name, f",
                         (arithmetic_unary | arithmetic_methods_unary).items())
@pytest.mark.parametrize('dtype', dtypes_numeric)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_arithmetic_unary(f_name, f, dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    res = f(marrays[0])
    ref = f(masked_arrays[0])
    assert_allclose(res, ref, seed=seed, xp=xp)


arithmetic_binary_exceptions = [
    "Integers to negative integer powers are not allowed.",
    "Only floating-point dtypes are allowed in __truediv__",
    "ufunc 'floor_divide' not supported for the input types",
    "ufunc 'remainder' not supported for the input types,",
    "Only real numeric dtypes are allowed in __floordiv__",
    "Only real numeric dtypes are allowed in __mod__",
    "of arguments for cupy_floor_divide",
    "of arguments for cupy_remainder",
    'ZeroDivisionError',  # torch
]


@pytest.mark.parametrize("f_name, f",
                         (arithmetic_binary | arithmetic_methods_binary).items())
@pytest.mark.parametrize('dtype', dtypes_numeric)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=arithmetic_binary_exceptions + torch_exceptions)
def test_arithmetic_binary(f_name, f, dtype, xp, seed=None):
    pass_backend(xp=xp, pass_xp='torch', dtype=dtype, pass_dtypes=['complex64'],
                 pass_funs=["x ** y"], pass_fun=pytest.xfail,
                 reason='Occasional tolerance issues.')
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    res = f(marrays[0], marrays[1])
    ref_data = f(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = masked_arrays[0].mask | masked_arrays[1].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_allclose(res, ref, seed=seed, xp=xp)


@pytest.mark.parametrize("f_name, f", (array_binary | array_methods_binary).items())
@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only numeric dtypes are allowed in matmul"]
                         + torch_exceptions)
def test_array_binary(f_name, f, dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, ndim=(2, 4), dtype=dtype, xp=xp, seed=seed)
    res = f(marrays[0], marrays[0].mT)
    x = masked_arrays[0].data
    mask = masked_arrays[0].mask
    x[mask] = 0
    data = f(x, x.mT.copy())  # .copy to prevent gh-33
    mask = ~f(~mask, ~mask.mT)
    ref = np.ma.masked_array(data, mask=mask)
    assert_allclose(res, ref, seed=seed, xp=xp, rtol=get_rtol(dtype, xp))


@pytest.mark.parametrize("f_name, f", (bitwise_unary | bitwise_methods_unary).items())
@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_bitwise_unary(f_name, f, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)

    res = f(~marrays[0])
    ref = f(~masked_arrays[0])
    assert_equal(res, ref, xp=xp, seed=seed)

    f = getattr(mxp, f_name)
    res = f(~marrays[0])
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("f_name, f", (bitwise_binary | bitwise_methods_binary).items())
@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["is only defined for x2 >= 0",
                          "Only integer dtypes are allowed in "]
                         + torch_exceptions
)
def test_bitwise_binary(f_name, f, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)

    res = f(marrays[0], marrays[1])
    ref = f(masked_arrays[0], masked_arrays[1])
    assert_equal(res, ref, xp=xp, seed=seed)

    f = getattr(mxp, f_name)
    res = f(marrays[0], marrays[1])
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize('type_val', scalar_conversions.items())
@pytest.mark.parametrize('mask', [False, True])
@pytest.mark.parametrize('xp', xps)
def test_scalar_conversion(type_val, mask, xp):
    mxp = marray.masked_namespace(xp)
    type, val = type_val
    x = mxp.asarray(val)
    assert type(x) == val
    assert isinstance(type(x), type)

    method = getattr(x, f"__{type.__name__}__")
    assert method() == val
    assert isinstance(method(), type)


@pytest.mark.parametrize('xp', xps)
def test_indexing(xp):
    # The implementations of `__getitem__` and `__setitem__` are trivial.
    # This does not make them easy to test exhaustively, but it does make
    # them easy to fix if a shortcoming is identified. Include a very basic
    # test for now, and improve as needed.
    mxp = marray.masked_namespace(xp)
    x = mxp.asarray(xp.arange(3), mask=[False, True, False])

    # Test `__setitem__`/`__getitem__` roundtrip
    x[1] = 10
    assert x[1] == 10
    assert isinstance(x[1], type(x))

    # Test `__setitem__`/`__getitem__` roundtrip with masked array as index
    i = mxp.asarray(1, mask=False)
    # CuPy doesn't have `__index__`. Check after release of CuPy 14.
    i_index = i.__index__() if hasattr(i.data, '__index__') else int(i)
    x[i_index] = 20
    assert x[i_index] == 20
    assert isinstance(x[i_index], type(x))

    # `__setitem__` can change mask
    x[1] = mxp.asarray(30, mask=False)
    assert x[1].data == 30
    assert x[1].mask == xp.asarray(False)
    x[2] = mxp.asarray(40, mask=True)
    assert x[2].data == 40
    assert x[2].mask == xp.asarray(True)

    # Indexing with masked array is not allowed
    message = "Correct behavior for indexing with a masked array..."
    with pytest.raises(NotImplementedError, match=message):
        x[x]
    with pytest.raises(NotImplementedError, match=message):
        x[x] = 1

    assert not np.iterable(xp.asarray(1))  # Check gh-72

    if xp == np:
        # Situation that came up in gh-71: key validation did not
        # consider that input could be a `tuple`. Some elements can
        # be MArrays that need to be validated/normalized.
        i = mxp.ones(4, dtype=mxp.bool)
        j = i[..., i]
        assert mxp.all(i == j)


@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize("xp", xps)
@pass_exceptions(allowed=["The maximum value of the data's dtype",
                          "has no attribute 'take_along_axis'",
                          "has no attribute '__array_namespace__'",
                          "Only real numeric dtypes are allowed in argsort"])
def test_take_along_axis(dtype, xp, seed=None):
    pass_backend(xp=xp, pass_xp='torch', dtype=dtype, pass_dtypes=dtypes_all)
    mxp = marray.masked_namespace(xp)
    marrays, _, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    x = marrays[0]
    i = mxp.argsort(x, axis=-1)
    res = mxp.take_along_axis(x, i, axis=-1)
    ref = mxp.sort(x, axis=-1)
    assert_equal(res, ref, xp=xp, seed=seed)

    rng = np.random.default_rng(seed)
    mask = xp.asarray(rng.random(i.shape) > 0.5)
    i = i.data
    i[mask] = 1000  # invalid index, but it will be masked
    i = mxp.asarray(i, mask=mask)
    res = mxp.take_along_axis(x, i, axis=-1)
    ref = mxp.asarray(ref.data, mask=(ref.mask | mask))
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["object has no attribute 'to_device'"])  # torch/cupy
def test_dlpack(dtype, xp, seed=None):
    # This is a placeholder for a real test when there is a real implementation
    mxp = marray.masked_namespace(xp)
    marrays, _, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    assert isinstance(marrays[0].__dlpack__(), type(marrays[0].data.__dlpack__()))
    assert marrays[0].__dlpack_device__() == marrays[0].data.__dlpack_device__()
    marrays[0].to_device(mxp.__array_namespace_info__().default_device())


@pytest.mark.parametrize("f_name, f",
                         (comparison_binary | comparison_methods_binary).items())
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only real numeric dtypes are allowed in"] + torch_exceptions)
def test_comparison_binary(f_name, f, dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    res = f(marrays[0], marrays[1])
    ref = f(masked_arrays[0], masked_arrays[1])
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("f", inplace_arithmetic + inplace_bitwise)
@pytest.mark.parametrize('arg2_masked', [True, False])
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_inplace(f, arg2_masked, dtype, xp, seed=None):
    pass_backend(xp=xp, pass_xp='torch', dtype=dtype, pass_dtypes=dtypes_integral)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    e1 = None
    e2 = None

    try:
        f(masked_arrays[0].data, masked_arrays[1].data)
        if arg2_masked:
            masked_arrays[0].mask |= masked_arrays[1].mask
        masked_arrays[0] = np.ma.masked_array(masked_arrays[0].data,
                                              masked_arrays[0].mask)
    except Exception as e:
        e1 = str(e)
    try:
        f(marrays[0], marrays[1] if arg2_masked else marrays[1].data)
    except Exception as e:
        e2 = str(e)

    # With one exception, either there is something wrong with both or the results agree
    if "Only numeric dtypes are allowed in" in str(e2):
        pass
    elif e1 or e2:
        assert e1 and e2
    else:
        assert_equal(marrays[0], masked_arrays[0], xp=xp, seed=seed)


@pytest.mark.parametrize("f", inplace_array)
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only numeric dtypes are allowed in matmul"])
def test_inplace_array_binary(f, dtype, xp, seed=None):
    pass_backend(xp=xp, pass_xp='torch', dtype=dtype,
                 pass_dtypes=['bool', 'uint16', 'uint32', 'uint64'])
    # very restrictive operator -> limited test
    mxp = marray.masked_namespace(xp)
    rng = np.random.default_rng(seed)
    data = (rng.random((3, 10, 10))*10).astype(dtype)
    mask = rng.random((3, 10, 10)) > 0.5
    a = mxp.asarray(xp.asarray(data, copy=True), mask=xp.asarray(mask, copy=True))
    data = (rng.random((3, 10, 10))*10).astype(dtype)
    mask = rng.random((3, 10, 10)) > 0.5
    b = mxp.asarray(xp.asarray(data, copy=True), mask=xp.asarray(mask, copy=True))
    ref = a @ b
    f(a, b)
    assert_allclose(a, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("f_name, f", arithmetic_binary.items())
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pytest.mark.parametrize('type_', ["array", "scalar"])
@pass_exceptions(allowed=["not supported for the input types",
                          "Only real numeric dtypes are allowed",
                          "Only numeric dtypes are allowed",
                          "Only floating-point dtypes are allowed",
                          "Integers to negative integer powers are not allowed",
                          "numpy boolean subtract, the `-` operator, is not supported",
                          "Subtraction, the `-` operator, with two bool tensors",
                          "Subtraction, the `-` operator, with a bool tensor",
                          "ZeroDivisionError"
                          ] + torch_exceptions)
def test_rarithmetic_binary(f_name, f, dtype, xp, type_, seed=None):
    pass_backend(xp=xp, pass_xp='torch', dtype=dtype, pass_dtypes=['complex64'],
                 pass_funs=["x ** y"], pass_fun=pytest.xfail,
                 reason='Occasional tolerance issues.')
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    if type_ == "array":
        arg1a = marrays[0].data
        arg1b = masked_arrays[0].data
    else:
        arg1a = arg1b = 2

    res = f(arg1a, marrays[1])
    ref_data = f(arg1b, masked_arrays[1].data)
    ref_mask = np.broadcast_to(masked_arrays[1].mask, ref_data.shape)
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_allclose(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only numeric dtypes are allowed in __matmul__"]
                         + torch_exceptions)
def test_rarray_binary(dtype, xp, seed=None):
    # very restrictive operator -> limited test
    mxp = marray.masked_namespace(xp)
    rng = np.random.default_rng(seed)
    data = (rng.random((3, 10, 10))*10).astype(dtype)
    mask = rng.random((3, 10, 10)) > 0.5
    a = mxp.asarray(xp.asarray(data, copy=True), mask=xp.asarray(mask, copy=True))
    data = (rng.random((3, 10, 10))*10).astype(dtype)
    mask = rng.random((3, 10, 10)) > 0.5
    b = mxp.asarray(xp.asarray(data, copy=True), mask=xp.asarray(mask, copy=True))
    res = a.data @ b
    ref = mxp.asarray(a.data) @ b
    assert_allclose(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("f", bitwise_binary.values())
@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only integer dtypes are allowed in"] + torch_exceptions)
def test_rbitwise_binary(f, dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    res = f(marrays[0].data, marrays[1])
    ref = f(masked_arrays[0].data, masked_arrays[1])
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
def test_attributes(dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, ndim=(2, 4), dtype=dtype, xp=xp, seed=seed)
    assert marrays[0].dtype == marrays[0].data.dtype
    assert marrays[0].device == marrays[0].data.device == marrays[0].mask.device
    assert xp.all(marrays[0].mT.data == marrays[0].data.mT)
    assert xp.all(marrays[0].mT.mask == marrays[0].mask.mT)
    assert marrays[0].ndim == marrays[0].data.ndim == marrays[0].mask.ndim
    assert marrays[0].shape == marrays[0].data.shape == marrays[0].mask.shape
    if "torch" not in str(xp):
        assert marrays[0].size == marrays[0].data.size == marrays[0].mask.size
    if marrays[0].ndim == 2:
        assert xp.all(marrays[0].T.data == marrays[0].data.T)
        assert xp.all(marrays[0].T.mask == marrays[0].mask.T)


@pytest.mark.parametrize('xp', xps)
def test_constants(xp):
    mxp = marray.masked_namespace(xp)
    assert mxp.e == xp.e
    assert mxp.inf == xp.inf
    assert np.isnan(mxp.nan) == np.isnan(xp.nan)
    assert mxp.newaxis == xp.newaxis
    assert mxp.pi == xp.pi


@pytest.mark.parametrize("f", dtypes_all + inspection + version + ['isdtype'])
@pytest.mark.parametrize('xp', xps)
def test_dtype_inspection_version(f, xp):
    mxp = marray.masked_namespace(xp)
    assert getattr(mxp, f) is getattr(xp, f)


@pytest.mark.parametrize("dtype", dtypes_real + dtypes_complex)
@pytest.mark.parametrize('xp', xps)
def test_finfo(dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, _, _ = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    dype_mxp, dtype_xp = getattr(mxp, dtype), getattr(xp, dtype)
    assert mxp.finfo(dype_mxp) == xp.finfo(dtype_xp)
    # see numpy/numpy#22977, data_apis/array_api_strict#116
    # assert mxp.finfo(marrays[0]) == xp.finfo(dtype)


@pytest.mark.parametrize("dtype", dtypes_integral)
@pytest.mark.parametrize('xp', xps)
def test_iinfo(dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, _, _ = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    # comparing the objects themselves doesn't work for NumPy
    dype_mxp, dtype_xp = getattr(mxp, dtype), getattr(xp, dtype)
    assert mxp.iinfo(dype_mxp).max == xp.iinfo(dtype_xp).max
    assert mxp.iinfo(dype_mxp).min == xp.iinfo(dtype_xp).min
    assert mxp.iinfo(dype_mxp).bits == xp.iinfo(dtype_xp).bits
    # see numpy/numpy#22977, data_apis/array_api_strict#116
    # assert mxp.iinfo(marrays[0]).max == xp.iinfo(dtype).max


@pytest.mark.parametrize('f_name', ['can_cast', 'result_type'])
@pytest.mark.parametrize('dtype1', dtypes_all)
@pytest.mark.parametrize('dtype2', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["cannot be type promoted together",
                          "Promotion for uint16, uint32, uint64 types"])
def test_can_cast_result_type(f_name, dtype1, dtype2, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    mxp_fun = getattr(mxp, f_name)
    xp_fun = getattr(xp, f_name)
    marrays, _, _ = get_arrays(1, dtype=dtype1, xp=xp, seed=seed)
    ref = xp_fun(getattr(xp, dtype1), getattr(xp, dtype2))
    if f_name == 'result_type' or xp == strict:  # see numpy/numpy#22977
        res1 = mxp_fun(marrays[0], getattr(mxp, dtype2))
        assert res1 == ref
    res2 = mxp_fun(getattr(mxp, dtype1), getattr(mxp, dtype2))
    assert res2 == ref


@pytest.mark.parametrize("f_name", elementwise_unary)
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["not supported for the input types",
                          "did not contain a loop with signature matching types",
                          "The numpy boolean negative, the `-` operator, is not supported",
                          "Only floating-point dtypes are allowed",
                          "Only real numeric dtypes are allowed",
                          "Only real floating-point dtypes are allowed",
                          "Only numeric dtypes are allowed",
                          "Only boolean dtypes are allowed",
                          "Only complex floating-point dtypes are allowed"]
                         + torch_exceptions)
def test_elementwise_unary(f_name, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    f = getattr(mxp, f_name)
    f2 = getattr(xp, f_name)
    res = f(marrays[0])
    ref_data = f2(xp.asarray(masked_arrays[0].data))
    ref_mask = masked_arrays[0].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("f_name", elementwise_binary)
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["not supported for the input types",
                          "numpy boolean subtract, the `-` operator, is not supported",
                          "Integers to negative integer powers are not allowed.",
                          "Only floating-point dtypes are allowed",
                          "Only real numeric dtypes are allowed",
                          "Only real floating-point dtypes are allowed",
                          "Only numeric dtypes are allowed",
                          "Only boolean dtypes are allowed",
                          "ZeroDivisionError"] + torch_exceptions)
def test_elementwise_binary(f_name, dtype, xp, seed=None):
    pass_backend(xp=xp, dtype=dtype, fun=f_name, pass_xp='torch',
                 pass_funs=['copysign', 'atan2'],
                 pass_dtypes=['bool', 'uint8', 'uint16', 'int8', 'int16'],
                 reason="Unexpected dtype", pass_fun=pytest.xfail)
    pass_backend(xp=xp, dtype=dtype, fun=f_name, pass_xp='torch', pass_fun=pytest.xfail,
                 pass_funs=['pow'], pass_dtypes=['complex64'], reason='Accuracy')
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    f = getattr(mxp, f_name)
    f2 = getattr(np, f_name)
    res = f(marrays[0], marrays[1])
    ref_data = f2(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = masked_arrays[0].mask | masked_arrays[1].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_allclose(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("f_name", ['any']) #statistical_array + utility_array + searching_array)
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only floating-point dtypes are allowed in __truediv__",
                          "Only numeric dtypes are allowed",
                          "Only real numeric dtypes are allowed"] + torch_exceptions)
def test_statistical_array(f_name, keepdims, xp, dtype, seed=None):
    if dtype.startswith('uint'):
        # should fix this and ensure strict check at the end
        pytest.skip("`np.ma` can't provide reference due to numpy/numpy#27885")

    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    rng = np.random.default_rng(seed)
    axes = list(range(marrays[0].ndim))
    axes = axes if f_name.startswith("cumulative_") else axes + [None]
    kwargs = {} if f_name.startswith("cumulative_") else {'keepdims': keepdims}
    f_map = {'cumulative_sum': 'cumsum', 'cumulative_prod': 'cumprod'}
    f_name2 = f_map.get(f_name, f_name)

    def _count_nonzero_ref(x, axis, keepdims):
        return np.sum((x.data != 0) & ~x.mask, axis=axis, keepdims=keepdims)

    axis = axes[rng.integers(len(axes))]
    f = getattr(mxp, f_name)
    f2 = getattr(np.ma, f_name2, _count_nonzero_ref)
    f3 = getattr(np, f_name2)
    res = f(marrays[0], axis=axis, **kwargs)
    ref = f2(masked_arrays[0], axis=axis, **kwargs)
    # masked array dtypes are not correct
    ref_dtype = np.asarray(f3(masked_arrays[0].data, axis=axis, **kwargs)).dtype

    # `argmin`/`argmax` don't calculate mask correctly
    ref_mask = np.all(masked_arrays[0].mask, axis=axis, **kwargs)
    ref = np.ma.masked_array(ref.data, getattr(ref, 'mask', ref_mask))
    ref = ref.astype(ref_dtype)
    assert_allclose(res, ref, xp=xp, seed=seed, strict=True, rtol=get_rtol(dtype, xp))

@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only numeric dtypes are allowed"] + torch_exceptions)
def test_cumulative_op_identity(dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, _, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    rng = np.random.default_rng(seed)
    axes = list(range(marrays[0].ndim))
    axis = axes[rng.integers(len(axes))]

    res = mxp.cumulative_sum(marrays[0], axis=axis, include_initial=True)
    identity = mxp.take(res, xp.asarray([0], dtype=xp.int64), axis=axis)
    assert not xp.any(identity.mask)
    assert xp.all(identity.data == 0)

    res = mxp.cumulative_prod(marrays[0], axis=axis, include_initial=True)
    identity = mxp.take(res, xp.asarray([0], dtype=xp.int64), axis=axis)
    assert not xp.any(identity.mask)
    assert xp.all(identity.data == 1)


@pass_exceptions(allowed=["Only numeric dtypes are allowed"])
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("prepend", [False, True])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only numeric dtypes are allowed in diff"] + torch_exceptions)
def test_diff(n, prepend, append, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    rng = np.random.default_rng(seed)
    marrays, masked_arrays, seed = get_arrays(3, dtype=dtype, xp=xp,
                                              pre_broadcasted=True, seed=seed)
    axes = list(range(marrays[0].ndim))
    axis = axes[rng.integers(len(axes))]
    kwargs_mxp = {'prepend': marrays[1] if prepend else None,
                  'append': marrays[2] if append else None}
    kwargs_np = {'prepend': masked_arrays[1]} if prepend else {}
    kwargs_np = kwargs_np | {'append': masked_arrays[2]} if append else kwargs_np
    res = mxp.diff(marrays[0], n=n, axis=axis, **kwargs_mxp)
    ref = np.ma.diff(masked_arrays[0], n=n, axis=axis, **kwargs_np)
    assert_allclose(res, ref, xp=xp, seed=seed, strict=True, rtol=get_rtol(dtype, xp))


# Test Creation functions
@pytest.mark.parametrize('f_name, args, kwargs', [
    # Try to pass options that change output compared to default
    ('arange', (1.5, 10, 2), dict()),
    ('asarray', ([1, 2, 3],), dict(copy=True)),
    ('empty', ((4, 3, 2),), dict()),
    ('empty_like', (np.empty((4, 3, 2)),), dict()),
    ('eye', (10, 11), dict(k=2)),
    ('full', ((4, 3, 2), 5), dict()),
    ('full_like', (np.empty((4, 3, 2)), 5.), dict()),
    ('linspace', (1, 20, 100), dict(endpoint=False)),
    ('ones', ((4, 3, 2),), dict()),
    ('ones_like', (np.empty((4, 3, 2)),), dict()),
    ('zeros', ((4, 3, 2),), dict()),
    ('zeros_like', (np.empty((4, 3, 2)),), dict()),
])
@pytest.mark.parametrize("dtype", dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=[r"arange() is only supported for booleans when"]
                         + torch_exceptions)
def test_creation(f_name, args, kwargs, dtype, xp, seed=None):
    dtype = getattr(xp, dtype)
    mxp = marray.masked_namespace(xp)
    f_xp = getattr(xp, f_name)
    f_mxp = getattr(mxp, f_name)
    if f_name.endswith('like'):
        args = tuple(xp.asarray(arg) for arg in args)
    res = f_mxp(*args, dtype=dtype, **kwargs)
    ref = f_xp(*args, dtype=dtype, **kwargs)
    if f_name.startswith('empty'):
        assert res.data.shape == ref.shape
    else:
        np.testing.assert_equal(res.data, np.asarray(ref), strict=True)
    np.testing.assert_equal(res.mask, np.full(ref.shape, False), strict=True)


@pytest.mark.parametrize('f_name',
                         ['empty_like', 'zeros_like', 'ones_like', 'full_like'])
@pytest.mark.parametrize("dtype", dtypes_all + [None])
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_creation_like(f_name, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    f_mxp = getattr(mxp, f_name)
    f_np = getattr(np, f_name)  # np.ma doesn't have full_like
    args = (2,) if f_name == "full_like" else ()
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    res = f_mxp(marrays[0], *args, dtype=getattr(xp, str(dtype), None))
    ref = f_np(masked_arrays[0], *args, dtype=dtype)
    if f_name.startswith('empty'):
        assert res.data.shape == ref.shape
        np.testing.assert_equal(res.mask, ref.mask)
    else:
        ref = np.ma.masked_array(ref, mask=masked_arrays[0].mask)
        assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize('f_name', ['tril', 'triu'])
@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_tri(f_name, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    f_xp = getattr(xp, f_name)
    f_mxp = getattr(mxp, f_name)
    marrays, _, seed = get_arrays(1, ndim=(2, 4), dtype=dtype, xp=xp, seed=seed)

    res = f_mxp(marrays[0], k=1)
    ref_data = f_xp(marrays[0].data, k=1)
    ref_mask = f_xp(marrays[0].mask, k=1)
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize('indexing', ['ij', 'xy'])
@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_meshgrid(indexing, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 4)
    marrays, masked_arrays, seed = get_arrays(n, ndim=1, dtype=dtype, xp=xp, seed=seed)

    res = mxp.meshgrid(*marrays, indexing=indexing)
    ref_data = np.meshgrid(*[array.data for array in masked_arrays], indexing=indexing)
    ref_mask = np.meshgrid(*[array.mask for array in masked_arrays], indexing=indexing)
    ref = [np.ma.masked_array(data, mask=mask)
           for data, mask in zip(ref_data, ref_mask)]
    for res_array, ref_array in zip(res, ref):
        assert_equal(res_array, ref_array, xp=xp, seed=seed)


@pytest.mark.parametrize("side", ['left', 'right'])
@pytest.mark.parametrize('dtype', dtypes_integral + dtypes_real)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_searchsorted(side, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)

    rng = np.random.default_rng(seed)
    n = 20
    m = 10

    x1 = rng.integers(10, size=n).astype(dtype)
    x1_mask = (rng.random(size=n) > 0.5)
    x2 = rng.integers(-2, 12, size=m).astype(dtype)
    x2_mask = rng.random(size=m) > 0.5

    x1 = mxp.asarray(x1, mask=x1_mask)
    x2 = mxp.asarray(x2, mask=x2_mask)

    # Note that the output of `searchsorted` is the same whether
    # a (valid) `sorter` is provided or the array is sorted to begin with
    res = xp.searchsorted(x1.data, x2.data, side=side, sorter=xp.argsort(x1.data))
    ref = xp.searchsorted(xp.sort(x1.data), x2.data, side=side, sorter=None)
    assert xp.all(res == ref)

    # This is true for `marray`, too
    res = mxp.searchsorted(x1, x2, side=side, sorter=mxp.argsort(x1))
    x1 = mxp.sort(x1)
    ref = mxp.searchsorted(x1, x2, side=side, sorter=None)
    assert mxp.all(res == ref)

    # And the output satisfies the required properties:
    for j in range(res.size):
        i = res[j]

        if i.mask:
            assert x2.mask[j]
            continue

        i = i.__index__()
        v = x2[j]
        if side == 'left':
            assert mxp.all(x1[:i] < v) and mxp.all(v <= x1[i:])
        else:
            assert mxp.all(x1[:i] <= v) and mxp.all(v < x1[i:])

@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_where(dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    rng = np.random.default_rng(seed)
    cond = rng.random(marrays[0].shape) > 0.5
    res = mxp.where(xp.asarray(cond), *marrays)
    ref = np.ma.where(cond, *masked_arrays)
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize('cond', [False, True])
@pytest.mark.parametrize('x1', [1, 1., 1+1j, np.int8(1), np.float32()])
@pytest.mark.parametrize('x2', [1, 1., 1+1j, np.int8(1), np.float32()])
def test_where_dtype(cond, x1, x2):
    # NumPy-only sanity check that result dtype is correct
    mxp = marray.masked_namespace(np)
    dtype = np.result_type(x1, x2)
    res = mxp.where(cond, x1, x2)
    assert res.dtype == dtype


@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_nonzero(dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    x, y = marrays[0], masked_arrays[0]
    rng = np.random.default_rng(seed)
    cond = rng.random(marrays[0].shape) > 0.5
    x[xp.asarray(cond)] = 0
    y[xp.asarray(cond)] = 0
    res = mxp.nonzero(x)
    ref = np.ma.nonzero(y)
    for i in range(len(ref)):
        np.testing.assert_equal(res[i].data, ref[i])
        np.testing.assert_equal(res[i].mask, np.full(ref[i].shape, False))


@pytest.mark.parametrize('f_name, n_arrays, n_dims, args, kwargs', [
    # Try to pass options that change output compared to default
    ('broadcast_arrays', 3, (3, 5), tuple(), dict()),
    ('broadcast_to', 1, (3, 5), tuple(), dict(shape=None)),
    ('concat', 3, (3, 5), tuple(), dict(axis=1)),
    ('expand_dims', 1, (3, 5), tuple(), dict(axis=1)),
    ('flip', 1, (3, 5), tuple(), dict(axis=1)),
    ('moveaxis', 1, (3, 5), (1, 2), dict()),
    ('permute_dims', 1, 3, tuple(), dict(axes=[2, 0, 1])),
    ('repeat', 1, (3, 5), (2,), dict(axis=1)),
    ('reshape', 1, (3, 5), tuple(), dict(shape=(-1,), copy=False)),
    ('roll', 1, (3, 5), tuple(), dict(shift=3, axis=1)),
    ('squeeze', 1, (3, 5), tuple(), dict(axis=1)),
    ('stack', 3, (3, 5), tuple(), dict(axis=1)),
    ('tile', 1, (3, 5), ((2, 3),), dict()),
    ('unstack', 1, (3, 5), tuple(), dict(axis=1)),
])
@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_manipulation(f_name, n_arrays, n_dims, args, kwargs, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, _, seed = get_arrays(n_arrays, ndim=n_dims, dtype=dtype, xp=xp, seed=seed)
    if f_name in {'broadcast_to', 'squeeze'}:
        original_shape = marrays[0].shape
        marrays[0] = marrays[0][:, 0:1, ...]
        if f_name == "broadcast_to":
            kwargs['shape'] = original_shape

    f_mxp = getattr(mxp, f_name)
    f_xp = getattr(xp, f_name)

    if f_name in {'concat', 'stack'}:
        marrays = mxp.broadcast_arrays(*marrays)
        res = (f_mxp(marrays, *args, **kwargs))
        ref_data = f_xp([marray.data for marray in marrays], *args, **kwargs)
        ref_mask = f_xp([marray.mask for marray in marrays], *args, **kwargs)
    else:
        res = f_mxp(*marrays, *args, **kwargs)
        ref_data = f_xp(*[marray.data for marray in marrays], *args, **kwargs)
        ref_mask = f_xp(*[marray.mask for marray in marrays], *args, **kwargs)

    ref = np.ma.masked_array(ref_data, mask=ref_mask)

    if f_name in {'broadcast_arrays', 'unstack'}:
        [assert_equal(res_i, ref_i, xp=xp, seed=seed) for res_i, ref_i in zip(res, ref)]
    else:
        assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.filterwarnings('ignore::numpy.exceptions.ComplexWarning')
@pytest.mark.filterwarnings('ignore:Casting complex values to real:UserWarning')
@pytest.mark.parametrize('dtype_in', dtypes_all)
@pytest.mark.parametrize('dtype_out', dtypes_all)
@pytest.mark.parametrize('copy', [False, True])
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["The Array API standard stipulates that casting"] + torch_exceptions)
def test_astype(dtype_in, dtype_out, copy, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype_in, xp=xp, seed=seed)

    res = mxp.astype(marrays[0], getattr(xp, dtype_out), copy=copy)
    if dtype_in == dtype_out:
        if copy:
            assert res.data is not marrays[0].data
            assert res.mask is not marrays[0].mask
        else:
            assert res.data is marrays[0].data
            assert res.mask is marrays[0].mask
    ref = masked_arrays[0].astype(dtype_out, copy=copy)
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize('xp', xps)
def test_asarray_device(xp):
    mxp = marray.masked_namespace(xp)
    message = "`device` argument is not implemented"
    with pytest.raises(NotImplementedError, match=message):
        mxp.asarray(xp.asarray([1, 2, 3]), device='coconut')


@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=["Only real numeric dtypes are allowed"] + torch_exceptions)
def test_clip(dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(3, dtype=dtype, xp=xp, seed=seed)
    min = mxp.minimum(marrays[1], marrays[2])
    max = mxp.maximum(marrays[1], marrays[2])
    res = mxp.clip(marrays[0], min=min, max=max)
    min = np.ma.masked_array(min.data, mask=min.mask)
    max = np.ma.masked_array(max.data, mask=max.mask)
    ref = np.ma.clip(masked_arrays[0], min, max)
    assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize("f_name", ['unique_values', 'unique_counts',
                                    'unique_inverse', 'unique_all'])
@pytest.mark.parametrize('dtype', dtypes_all)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_set(f_name, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, _, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    f_mxp = getattr(mxp, f_name)

    if not xp.any(marrays[0].mask):
        pytest.skip("Tested by array_api_tests.")

    x = marrays[0]
    data = x.data
    mask = x.mask
    sentinel = marray._xinfo(x).max

    if mxp.any(x == sentinel):
        message = "The maximum value of the data's dtype is included"
        with pytest.raises(NotImplementedError, match=message):
            f_mxp(x)
        return

    res = f_mxp(x)

    data[mask] = sentinel
    ref = np.unique_all(np.asarray(data))
    ref_mask = np.asarray((ref.values == sentinel))

    ref_values = np.ma.masked_array(np.asarray(ref.values), mask=ref_mask)
    res_values = res if f_name == "unique_values" else res.values
    assert_equal(res_values, ref_values, xp=xp, seed=seed)

    if hasattr(res, 'counts'):
        ref_counts = np.ma.masked_array(np.asarray(ref.counts), mask=ref_mask)
        assert_equal(res.counts, ref_counts, xp=xp, seed=seed)

    if hasattr(res, 'indices'):
        ref_counts = np.ma.masked_array(np.asarray(ref.indices), mask=ref_mask)
        assert_equal(res.indices, ref_counts, xp=xp, seed=seed)

    if hasattr(res, 'inverse_indices'):
        ref_counts = np.ma.masked_array(np.asarray(ref.inverse_indices), mask=False)
        assert_equal(res.inverse_indices, ref_counts, xp=xp, seed=seed)


@pytest.mark.parametrize("f_name", ['sort', 'argsort'])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize('dtype', dtypes_real + dtypes_integral)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=torch_exceptions)
def test_sorting(f_name, descending, stable, dtype, xp, seed=None):
    mxp = marray.masked_namespace(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    f_mxp = getattr(mxp, f_name)
    f_xp = getattr(np.ma, f_name)

    info = marray._xinfo(marrays[0])
    sentinel = info.min if descending else info.max
    if mxp.any(marrays[0] == sentinel) and xp.any(marrays[0].mask):
        message = "value of the data's dtype is included"
        with pytest.raises(NotImplementedError, match=message):
            f_mxp(marrays[0], axis=-1, descending=descending, stable=stable)
        return

    if descending and xp==np:
        pytest.skip("NumPy doesn't have `descending`.")

    res = f_mxp(marrays[0], axis=-1, descending=descending, stable=stable)

    if stable:
        pytest.skip("No easy reference for `stable=True`.")

    if descending and dtype in dtypes_uint:
        pytest.skip("No easy reference for unsigned int with `descending=True`.")

    if descending:
        ref = f_xp(-masked_arrays[0], axis=-1, stable=stable)
        ref = -ref if f_name=='sort' else ref
    else:
        ref = f_xp(masked_arrays[0], axis=-1, stable=stable)

    if f_name == 'sort':
        assert_equal(res, np.ma.masked_array(ref), xp=xp, seed=seed)
    else:
        # We can't just compare the indices because sometimes `np.ma.argsort`
        # doesn't sort the masked elements the same way. Instead, we use the
        # indices to sort the arrays, then compare the sorted masked arrays.
        # (The difference is that we don't compare the masked values.)
        i_sorted = np.asarray(res.data)
        res_data = np.take_along_axis(np.asarray(marrays[0].data), i_sorted, axis=-1)
        res_mask = np.take_along_axis(np.asarray(marrays[0].mask), i_sorted, axis=-1)
        res = mxp.asarray(res_data, mask=res_mask)
        ref_data = np.take_along_axis(masked_arrays[0].data, ref, axis=-1)
        ref_mask = np.take_along_axis(masked_arrays[0].mask, ref, axis=-1)
        ref = np.ma.masked_array(ref_data, mask=ref_mask)
        assert_equal(res, ref, xp=xp, seed=seed)


@pytest.mark.parametrize('xp', xps)
def test_array_namespace(xp):
    mxp = marray.masked_namespace(xp)
    x = mxp.asarray([1, 2, 3])
    assert x.__array_namespace__() is mxp
    assert x.__array_namespace__("2024.12") is mxp
    message = "MArray interface for Array API version 'shrubbery'..."
    with pytest.raises(NotImplementedError, match=message):
        x.__array_namespace__("shrubbery")


def test_import():
    from marray import numpy as mnp
    assert mnp.__name__ == 'marray.numpy'
    from marray.numpy import asarray
    asarray(10, mask=True)

    from marray import array_api_strict as mxp
    assert mxp.__name__ == 'marray.array_api_strict'
    from marray.array_api_strict import asarray
    asarray(10, mask=True)


@pytest.mark.parametrize('xp', xps)
def test_str(xp):
    mxp = marray.masked_namespace(xp)
    x = mxp.asarray(1, mask=True)
    ref = "_"
    assert str(x) == ref


def test_repr():
    mxp = marray.masked_namespace(strict)
    x = mxp.asarray(1, mask=True)
    ref = ('MArray(\n    Array(_, dtype=array_api_strict.int64),'
           '\n    Array(True, dtype=array_api_strict.bool)\n)')
    assert repr(x) == ref

    mxp = marray.masked_namespace(np)
    x = mxp.asarray(1, mask=True)
    ref = "MArray(array(_), array(True))"
    assert repr(x) == ref


def test_signature_docs():
    # Rough test that signatures were replaced where possible
    mxp = marray.masked_namespace(np)
    assert mxp.sum.__signature__ == inspect.signature(np.sum)
    assert np.sum.__doc__ in mxp.sum.__doc__

# To do:
# - investigate asarray - is copy respected?

### Bug-fix tests

def test_gh33():
    # See https://github.com/mdhaber/marray/issues/33
    test_array_binary(*list(array_binary.items())[0], dtype='float32', xp=np, seed=566)


@pytest.mark.parametrize('xp', xps)
def test_gh99(xp):
    # https://github.com/mdhaber/marray/issues/99
    mxp = marray.masked_namespace(xp)
    assert mxp.any(mxp.asarray(0)) == False
    assert mxp.any(mxp.asarray(1)) == True


def test_test():
    # dev tool to reproduce a particular failure of a `parametrize`d test
    seed = 99154806044603893677768861545372495022
    # f_name, descending, stable, dtype, xp,
    test_set('unique_all', dtype='uint64', xp=torch, seed=seed)
