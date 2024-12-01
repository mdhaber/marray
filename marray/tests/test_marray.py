import functools
import itertools
import operator

import array_api_strict as strict
import numpy as np
import pytest

import marray

xps = [np, strict]
dtypes_boolean = ['bool']
dtypes_uint = ['uint8', 'uint16', 'uint32', 'uint64', ]
dtypes_int = ['int8', 'int16', 'int32', 'int64']
dtypes_real = ['float32', 'float64']
dtypes_complex = ['complex64', 'complex128']
dtypes_integral = dtypes_uint + dtypes_int
dtypes_numeric = dtypes_integral + dtypes_real + dtypes_complex
dtypes_all = dtypes_boolean + dtypes_integral + dtypes_real + dtypes_complex


def get_arrays(n_arrays, *, ndim=(1, 4), dtype='float64', xp=np, seed=None):
    xpm = marray.masked_array(xp)

    entropy = np.random.SeedSequence(seed).entropy
    rng = np.random.default_rng(entropy)

    ndim = rng.integers(*ndim) if isinstance(ndim, tuple) else ndim
    shape = rng.integers(1, 20, size=ndim)

    datas = []
    masks = []
    for i in range(n_arrays):
        shape_mask = rng.random(size=ndim) > 0.85
        shape_i = shape.copy()
        shape_i[shape_mask] = 1
        data = rng.standard_normal(size=shape_i)

        if dtype == 'bool':
            data = data > 0
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
    if xp is not None:
        array_type = type(xp.asarray(1.))
        assert isinstance(res.data, array_type)
        assert isinstance(res.mask, array_type)
    ref_mask = np.broadcast_to(ref.mask, ref.data.shape)
    try:
        comparison(res.data[~res.mask], ref.data[~ref_mask], strict=True, **kwargs)
        comparison(res.mask, ref_mask, strict=True, **kwargs)
    except AssertionError as e:
        raise AssertionError(seed) from e


def assert_equal(res, ref, seed, xp=None, **kwargs):
    return assert_comparison(res, ref, seed, xp, np.testing.assert_equal, **kwargs)


def assert_allclose(res, ref, seed, xp=None, **kwargs):
    return assert_comparison(res, ref, seed, xp, np.testing.assert_allclose, **kwargs)


def pass_exceptions(allowed=[]):
    def outer(f):
        @functools.wraps(f)
        def inner(*args, seed=None, **kwargs):
            try:
                return f(*args, seed=seed, **kwargs)
            except (ValueError, TypeError) as e:
                for message in allowed:
                    if str(e).startswith(message):
                        return
                else:
                    raise AssertionError(seed) from e
        return inner
    return outer


def get_rtol(dtype, xp):
    if isinstance(dtype, str):
        dtype = getattr(xp, dtype)
    if xp.isdtype(dtype, ('real floating', 'complex floating')):
        return xp.finfo(dtype).eps**0.5
    else:
        return 0


arithmetic_unary = [lambda x: +x, lambda x: -x, abs]
arithmetic_methods_unary = [lambda x: x.__abs__(), lambda x: x.__neg__(),
                            lambda x: x.__pos__()]
arithmetic_binary = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y,
                     lambda x, y: x / y, lambda x, y: x // y, lambda x, y: x % y,
                     lambda x, y: x ** y]
arithmetic_methods_binary = [lambda x, y: x.__add__(y), lambda x, y: x.__floordiv__(y),
                             lambda x, y: x.__mod__(y), lambda x, y: x.__mul__(y),
                             lambda x, y: x.__pow__(y), lambda x, y: x.__sub__(y),
                             lambda x, y: x.__truediv__(y)]
array_binary = [lambda x, y: x @ y, operator.matmul, operator.__matmul__]
array_methods_binary = [lambda x, y: x.__matmul__(y)]
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

comparison_binary = [lambda x, y: x < y, lambda x, y: x <= y, lambda x, y: x > y,
                     lambda x, y: x >= y, lambda x, y: x == y , lambda x, y: x != y]
comparison_methods_binary = [lambda x, y: x.__eq__(y), lambda x, y: x.__ge__(y),
                             lambda x, y: x.__gt__(y), lambda x, y: x.__le__(y),
                             lambda x, y: x.__lt__(y), lambda x, y: x.__ne__(y)]

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


data_type = ['can_cast', 'finfo', 'iinfo', 'isdtype']
inspection = ['__array_namespace_info__']
version = ['__array_api_version__']
elementwise_unary = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
                     'ceil', 'cos', 'cosh', 'exp', 'expm1', 'floor', 'isfinite',
                     'isinf', 'isnan', 'log', 'log1p', 'log2', 'log10',
                     'logical_not', 'negative', 'positive', 'round', 'sign',
                     'signbit', 'sin', 'sinh', 'square', 'sqrt', 'tan', 'tanh',
                     'trunc']
elementwise_unary_complex = ['real', 'imag', 'conj']
elementwise_binary = ['add', 'atan2', 'copysign', 'divide', 'equal', 'floor_divide',
                      'greater', 'greater_equal', 'hypot', 'less', 'less_equal',
                      'logaddexp', 'logical_and', 'logical_or', 'logical_xor',
                      'maximum', 'minimum', 'multiply', 'not_equal', 'pow',
                      'remainder', 'subtract']
searching_array = ['argmax', 'argmin']
statistical_array = ['cumulative_sum', 'max', 'mean',
                     'min', 'prod', 'std', 'sum', 'var']
utility_array = ['all', 'any']


@pytest.mark.parametrize("f", arithmetic_unary[:1] + arithmetic_methods_unary)
@pytest.mark.parametrize('dtype', dtypes_numeric)
@pytest.mark.parametrize('xp', xps)
def test_arithmetic_unary(f, dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, xp=xp, seed=seed)
    res = f(marrays[0])
    ref = f(masked_arrays[0])
    assert_equal(res, ref, seed=seed, xp=xp)


arithetic_binary_exceptions = [
    "Integers to negative integer powers are not allowed.",
    "Only floating-point dtypes are allowed in __truediv__",
    "ufunc 'floor_divide' not supported for the input types",
    "ufunc 'remainder' not supported for the input types,",
    "Only real numeric dtypes are allowed in __floordiv__",
    "Only real numeric dtypes are allowed in __mod__"
]


@pytest.mark.parametrize("f", arithmetic_binary + arithmetic_methods_binary)
@pytest.mark.parametrize('dtype', dtypes_numeric)
@pytest.mark.parametrize('xp', xps)
@pass_exceptions(allowed=arithetic_binary_exceptions)
def test_arithmetic_binary(f, dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, xp=xp, seed=seed)
    res = f(marrays[0], marrays[1])
    ref_data = f(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = masked_arrays[0].mask | masked_arrays[1].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed=seed, xp=xp)


@pytest.mark.parametrize('xp', xps)
@pytest.mark.parametrize("f", array_binary + array_methods_binary)
@pytest.mark.parametrize('dtype', dtypes_all)
@pass_exceptions(allowed=["Only numeric dtypes are allowed in matmul"])
def test_array_binary(f, dtype, xp, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, ndim=(2, 4), xp=xp, dtype=dtype, seed=seed)
    res = f(marrays[0], marrays[0].mT)
    x = masked_arrays[0].data
    mask = masked_arrays[0].mask
    x[mask] = 0
    data = f(x, x.mT)
    mask = ~f(~mask, ~mask.mT)
    ref = np.ma.masked_array(data, mask=mask)
    assert_allclose(res, ref, seed=seed, xp=xp, rtol=get_rtol(dtype, xp))


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize("f_name_fun", itertools.chain(bitwise_unary.items(),
                                                       bitwise_methods_unary.items()))
def test_bitwise_unary(f_name_fun, dtype, xp=np, seed=None):
    f_name, f = f_name_fun
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, seed=seed)

    res = f(~marrays[0])
    ref = f(~masked_arrays[0])
    assert_equal(res, ref, seed)

    f = getattr(mxp, f_name)
    res = f(~marrays[0])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize("f_name_fun", itertools.chain(bitwise_binary.items(),
                                                       bitwise_methods_binary.items()))
def test_bitwise_binary(f_name_fun, dtype, xp=np, seed=None):
    f_name, f = f_name_fun
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)

    res = f(marrays[0], marrays[1])
    ref = f(masked_arrays[0], masked_arrays[1])
    assert_equal(res, ref, seed)

    f = getattr(mxp, f_name)
    res = f(marrays[0], marrays[1])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize('type_val', scalar_conversions.items())
@pytest.mark.parametrize('mask', [False, True])
def test_scalar_conversion(type_val, mask, xp=np):
    mxp = marray.masked_array(xp)
    type, val = type_val
    x = mxp.asarray(val)
    assert type(x) == val
    assert isinstance(type(x), type)

    method = getattr(x, f"__{type.__name__}__")
    assert method() == val
    assert isinstance(method(), type)


def test_indexing(xp=strict):
    # The implementations of `__getitem__` and `__setitem__` are trivial.
    # This does not make them easy to test exhaustively, but it does make
    # them easy to fix if a shortcoming is identified. Include a very basic
    # test for now, and improve as needed.
    mxp = marray.masked_array(xp)
    x = mxp.asarray(xp.arange(3), mask=[False, True, False])

    # Test `__setitem__`/`__getitem__` roundtrip
    x[1] = 10
    assert x[1] == 10
    assert isinstance(x[1], type(x))

    # Test `__setitem__`/`__getitem__` roundtrip with masked array as index
    i = mxp.asarray(1, mask=True)
    x[i.__index__()] = 20
    assert x[i.__index__()] == 20
    assert isinstance(x[i.__index__()], type(x))

    # `__setitem__` can change mask
    x[1] = mxp.asarray(30, mask=False)
    assert x[1].data == 30
    assert x[1].mask == False
    x[2] = mxp.asarray(40, mask=True)
    assert x[2].data == 40
    assert x[2].mask == True


def test_dlpack(xp=strict, seed=None):
    # This is a placeholder for a real test when there is a real implementation
    mxp = marray.masked_array(xp)
    marrays, _, seed = get_arrays(1, seed=seed)
    assert isinstance(marrays[0].__dlpack__(), type(marrays[0].data.__dlpack__()))
    assert marrays[0].__dlpack_device__() == marrays[0].data.__dlpack_device__()
    marrays[0].to_device('cpu')


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_real)
@pytest.mark.parametrize("f", comparison_binary + comparison_methods_binary)
def test_comparison_binary(f, dtype, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
    res = f(marrays[0], marrays[1])
    ref = f(masked_arrays[0], masked_arrays[1])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_real)
@pytest.mark.parametrize("f", inplace_arithmetic + inplace_bitwise)
def test_inplace(f, dtype, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
    e1 = None
    e2 = None

    try:
        f(masked_arrays[0].data, masked_arrays[1].data)
        masked_arrays[0].mask |= masked_arrays[1].mask
        masked_arrays[0] = np.ma.masked_array(masked_arrays[0].data,
                                              masked_arrays[0].mask)
    except Exception as e:
        e1 = str(e)
    try:
        f(marrays[0], marrays[1])
    except Exception as e:
        e2 = str(e)

    # either there is something wrong with both or the two results agree
    if e1 or e2:
        assert e1 and e2
    else:
        assert_equal(marrays[0], masked_arrays[0], seed)


@pytest.mark.parametrize("f", inplace_array)
def test_inplace_array_binary(f, xp=np, seed=None):
    # very restrictive operator -> limited test
    mxp = marray.masked_array(xp)
    rng = np.random.default_rng(seed)
    data = rng.random((3, 10, 10))
    mask = rng.random((3, 10, 10)) > 0.5
    a = mxp.asarray(data.copy(), mask=mask.copy())
    data = rng.random((3, 10, 10))
    mask = rng.random((3, 10, 10)) > 0.5
    b = mxp.asarray(data.copy(), mask=mask.copy())
    ref = a @ b
    f(a, b)
    assert_allclose(a, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_real)
@pytest.mark.parametrize("f", arithmetic_binary)
def test_rarithmetic_binary(f, dtype, seed=None):
    mxp = marray.masked_array(strict)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
    marrays[0] = mxp.asarray(marrays[0].data, mask=marrays[0].mask)
    marrays[1] = mxp.asarray(marrays[1].data, mask=marrays[1].mask)

    res = f(marrays[0].data, marrays[1])
    ref_data = f(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = np.broadcast_to(masked_arrays[1].mask, ref_data.shape)
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed)

    # Check that reflected operator works with Python scalar
    res = f(2, marrays[0])
    ref = f(getattr(np, dtype)(2), masked_arrays[0])
    assert_equal(res, ref, seed)


def test_rarray_binary(xp=np, seed=None):
    # very restrictive operator -> limited test
    mxp = marray.masked_array(strict)
    rng = np.random.default_rng(seed)
    data = rng.random((3, 10, 10))
    mask = rng.random((3, 10, 10)) > 0.5
    a = mxp.asarray(data, mask=mask)
    data = rng.random((3, 10, 10))
    mask = rng.random((3, 10, 10)) > 0.5
    b = mxp.asarray(data.copy(), mask=mask.copy())
    res = a.data @ b
    ref = mxp.asarray(a.data) @ b
    np.testing.assert_equal(np.asarray(res.data), np.asarray(ref.data))
    np.testing.assert_equal(np.asarray(res.mask), np.asarray(ref.mask))


@pytest.mark.parametrize("dtype", dtypes_integral)
@pytest.mark.parametrize("f", bitwise_binary.values())
def test_rbitwise_binary(f, dtype, seed=None):
    mxp = marray.masked_array(strict)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
    marrays[0] = mxp.asarray(marrays[0].data, mask=marrays[0].mask)
    marrays[1] = mxp.asarray(marrays[1].data, mask=marrays[1].mask)

    res = f(marrays[0].data, marrays[1])
    ref = f(masked_arrays[0].data, masked_arrays[1])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_all)
def test_attributes(dtype, seed=None, xp=np):
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, seed=seed)
    assert marrays[0].dtype == marrays[0].data.dtype
    assert marrays[0].device == marrays[0].data.device == marrays[0].mask.device
    if marrays[0].ndim >= 2:
        assert xp.all(marrays[0].mT.data == marrays[0].data.mT)
        assert xp.all(marrays[0].mT.mask == marrays[0].mask.mT)
    assert marrays[0].ndim == marrays[0].data.ndim == marrays[0].mask.ndim
    assert marrays[0].shape == marrays[0].data.shape == marrays[0].mask.shape
    assert marrays[0].size == marrays[0].data.size == marrays[0].mask.size
    assert xp.all(marrays[0].T.data == marrays[0].data.T)
    assert xp.all(marrays[0].T.mask == marrays[0].mask.T)


def test_constants(xp=np):
    mxp = marray.masked_array(xp)
    assert mxp.e == xp.e
    assert mxp.inf == xp.inf
    assert np.isnan(mxp.nan) == np.isnan(xp.nan)
    assert mxp.newaxis == xp.newaxis
    assert mxp.pi == xp.pi


@pytest.mark.parametrize("f", data_type + inspection + version)
def test_dtype_funcs_inspection(f, xp=strict):
    mxp = marray.masked_array(xp)
    getattr(mxp, f) is getattr(xp, f)


@pytest.mark.parametrize("dtype", dtypes_all)
def test_dtypes(dtype, xp=strict):
    # NumPy fails... unclear whether xp.bool must be a "dtype"
    mxp = marray.masked_array(xp)
    getattr(mxp, dtype).__eq__(getattr(xp, dtype))


@pytest.mark.parametrize("f_name", elementwise_unary)
def test_elementwise_unary(f_name, xp=np, dtype='float64', seed=None):
    # TODO: confirm that NaNs should not automatically get masked
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, seed=seed)
    f = getattr(mxp, f_name)
    f2 = getattr(xp, f_name)
    res = f(marrays[0])
    ref_data = f2(masked_arrays[0].data)
    ref_mask = masked_arrays[0].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("f_name", elementwise_unary_complex)
@pytest.mark.parametrize('dtype', dtypes_complex)
def test_elementwise_unary_complex(f_name, dtype, seed=None):
    test_elementwise_unary(f_name, dtype=dtype, seed=seed)


@pytest.mark.parametrize("f_name", elementwise_binary)
def test_elementwise_binary(f_name, xp=np, dtype='float64', seed=None):
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
    f = getattr(mxp, f_name)
    f2 = getattr(xp, f_name)
    res = f(marrays[0], marrays[1])
    ref_data = f2(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = masked_arrays[0].mask | masked_arrays[1].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("f_name", statistical_array + utility_array + searching_array)
def test_statistical_array(f_name, keepdims, xp=np, dtype='float64', seed=None):
    # TODO: confirm that result should never have mask? Only when all are masked?
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, seed=seed)
    rng = np.random.default_rng(seed)
    axes = list(range(marrays[0].ndim))
    axes = axes if f_name == "cumulative_sum" else axes + [None]
    kwargs = {} if f_name == "cumulative_sum" else {'keepdims': keepdims}
    f_name2 = 'cumsum' if f_name == "cumulative_sum" else f_name

    axis = axes[rng.integers(len(axes))]
    f = getattr(mxp, f_name)
    f2 = getattr(xp, f_name2)
    res = f(marrays[0], axis=axis, **kwargs)
    ref = f2(masked_arrays[0], axis=axis, **kwargs)
    # `argmin`/`argmax` don't calculate mask correctly
    ref_mask = np.all(masked_arrays[0].mask, axis=axis, **kwargs)
    ref = np.ma.masked_array(ref.data, getattr(ref, 'mask', ref_mask))
    assert_equal(res, ref, seed)


# Test Creation functions
@pytest.mark.parametrize('f_name, args, kwargs', [
    # Try to pass options that change output compared to default
    ('arange', (-1.5, 10, 2), dict(dtype=int)),
    ('asarray', ([1, 2, 3],), dict(dtype=float, copy=True)),
    ('empty', ((4, 3, 2),), dict(dtype=int)),
    ('empty_like', (np.empty((4, 3, 2)),), dict(dtype=int)),
    ('eye', (10, 11), dict(k=2, dtype=int)),
    ('full', ((4, 3, 2), 5), dict(dtype=float)),
    ('full_like', (np.empty((4, 3, 2)), 5.), dict(dtype=int)),
    ('linspace', (1, 20, 100), dict(dtype=int, endpoint=False)),
    ('ones', ((4, 3, 2),), dict(dtype=int)),
    ('ones_like', (np.empty((4, 3, 2)),), dict(dtype=int)),
    ('zeros', ((4, 3, 2),), dict(dtype=int)),
    ('zeros_like', (np.empty((4, 3, 2)),), dict(dtype=int)),
])
# Should `_like` functions inherit the mask of the argument?
def test_creation(f_name, args, kwargs, xp=np):
    mxp = marray.masked_array(xp)
    f_xp = getattr(xp, f_name)
    f_mxp = getattr(mxp, f_name)
    res = f_mxp(*args, **kwargs)
    ref = f_xp(*args, **kwargs)
    if f_name.startswith('empty'):
        assert res.data.shape == ref.shape
    else:
        np.testing.assert_equal(res.data, ref, strict=True)
    np.testing.assert_equal(res.mask, xp.full(ref.shape, False), strict=True)


@pytest.mark.parametrize('f_name', ['tril', 'triu'])
@pytest.mark.parametrize('dtype', dtypes_all)
def test_tri(f_name, dtype, seed=None, xp=np):
    mxp = marray.masked_array(xp)
    f_xp = getattr(xp, f_name)
    f_mxp = getattr(mxp, f_name)
    marrays, _, seed = get_arrays(1, ndim=(2, 4), dtype=dtype, seed=seed)

    res = f_mxp(marrays[0], k=1)
    ref_data = f_xp(marrays[0].data, k=1)
    ref_mask = f_xp(marrays[0].mask, k=1)
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed)


@pytest.mark.parametrize('indexing', ['ij', 'xy'])
@pytest.mark.parametrize('dtype', dtypes_all)
def test_meshgrid(indexing, dtype, seed=None, xp=np):
    mxp = marray.masked_array(xp)
    marrays, _, seed = get_arrays(1, ndim=1, dtype=dtype, seed=seed)

    res = mxp.meshgrid(*marrays, indexing=indexing)
    ref_data = xp.meshgrid([marray.data for marray in marrays], indexing=indexing)
    ref_mask = xp.meshgrid([marray.mask for marray in marrays], indexing=indexing)
    ref = [np.ma.masked_array(data, mask=mask) for data, mask in zip(ref_data, ref_mask)]
    [assert_equal(res_array, ref_array, seed) for res_array, ref_array in zip(res, ref)]


@pytest.mark.parametrize("side", ['left', 'right'])
def test_searchsorted(side, xp=strict, seed=None):
    mxp = marray.masked_array(xp)

    rng = np.random.default_rng(seed)
    n = 20
    m = 10

    x1 = rng.integers(10, size=n)
    x1_mask = (rng.random(size=n) > 0.5)
    x2 = rng.integers(-2, 12, size=m)
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


# Test Linear Algebra functions

# Use Array API tests to test the following:
# Creation Functions (same behavior but with all-False mask)
# Indexing (same behavior as indexing data and mask separately)


@pytest.mark.parametrize('f_name, n_arrays, n_dims, kwargs', [
    # Try to pass options that change output compared to default
    ('broadcast_arrays', 3, (3, 5), dict()),
    ('broadcast_to', 1, (3, 5), dict(shape=None)),
    ('concat', 3, (3, 5), dict(axis=1)),
    ('expand_dims', 1, (3, 5), dict(axis=1)),
    ('flip', 1, (3, 5), dict(axis=1)),
    ('moveaxis', 1, (3, 5), dict(source=1, destination=2)),
    ('permute_dims', 1, 3, dict(axes=[2, 0, 1])),
    ('repeat', 1, (3, 5), dict(repeats=2, axis=1)),
    ('reshape', 1, (3, 5), dict(shape=(-1,), copy=False)),
    ('roll', 1, (3, 5), dict(shift=3, axis=1)),
    ('squeeze', 1, (3, 5), dict(axis=1)),
    ('stack', 3, (3, 5), dict(axis=1)),
    ('tile', 1, (3, 5), dict(reps=(2, 3))),
    ('unstack', 1, (3, 5), dict(axis=1)),
])
def test_creation(f_name, n_arrays, n_dims, kwargs, seed=None, xp=np):
    mxp = marray.masked_array(xp)
    marrays, _, seed = get_arrays(n_arrays, ndim=n_dims, dtype=xp.float64, seed=seed)
    if f_name in {'broadcast_to', 'squeeze'}:
        original_shape = marrays[0].shape
        marrays[0] = marrays[0][:, 0:1, ...]
        if f_name == "broadcast_to":
            kwargs['shape'] = original_shape

    f_mxp = getattr(mxp, f_name)
    f_xp = getattr(xp, f_name)

    if f_name in {'concat', 'stack'}:
        marrays = mxp.broadcast_arrays(*marrays)
        res = (f_mxp(marrays, **kwargs))
        ref_data = f_xp([marray.data for marray in marrays], **kwargs)
        ref_mask = f_xp([marray.mask for marray in marrays], **kwargs)
    else:
        res = f_mxp(*marrays, **kwargs)
        ref_data = f_xp(*[marray.data for marray in marrays], **kwargs)
        ref_mask = f_xp(*[marray.mask for marray in marrays], **kwargs)

    ref = np.ma.masked_array(ref_data, mask=ref_mask)

    if f_name in {'broadcast_arrays', 'unstack'}:
        [assert_equal(res_i, ref_i, seed) for res_i, ref_i in zip(res, ref)]
    else:
        assert_equal(res, ref, seed)


@pytest.mark.filterwarnings('ignore::numpy.exceptions.ComplexWarning')
@pytest.mark.parametrize('dtype_in', dtypes_all)
@pytest.mark.parametrize('dtype_out', dtypes_all)
@pytest.mark.parametrize('copy', [False, True])
def test_astype(dtype_in, dtype_out, copy, xp=np, seed=None):
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype_in, seed=seed)

    res = mxp.astype(marrays[0], dtype_out, copy=copy)
    if dtype_in == dtype_out:
        if copy:
            assert res.data is not marrays[0].data
            assert res.mask is not marrays[0].mask
        else:
            assert res.data is marrays[0].data
            assert res.mask is marrays[0].mask
    ref = masked_arrays[0].astype(dtype_out, copy=copy)
    assert_equal(res, ref, seed)


@pytest.mark.parametrize('dtype', dtypes_real)
def test_clip(dtype, xp=np, seed=None):
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(3, dtype=dtype, seed=seed)
    res = mxp.clip(marrays[0], min=marrays[1], max=marrays[2])
    ref = np.ma.clip(*masked_arrays)
    assert_equal(res, ref, seed)

#?
# Set functions
# __array_namespace__


@pytest.mark.parametrize("f_name", ['sort', 'argsort'])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("stable", [False])  # NumPy masked arrays don't support True
@pytest.mark.parametrize('dtype', dtypes_real)
def test_sorting(f_name, descending, stable, dtype, xp=strict, seed=None):
    mxp = marray.masked_array(xp)
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, seed=seed)
    f_mxp = getattr(mxp, f_name)
    f_xp = getattr(np.ma, f_name)
    res = f_mxp(marrays[0], axis=-1, descending=descending, stable=stable)
    if descending:
        ref = f_xp(-masked_arrays[0], axis=-1, stable=stable)
        ref = -ref if f_name=='sort' else ref
    else:
        ref = f_xp(masked_arrays[0], axis=-1, stable=stable)

    if f_name == 'sort':
        assert_equal(res, np.ma.masked_array(ref), seed)
    else:
        # We can't just compare the indices because sometimes `np.ma.argsort`
        # doesn't sort the masked elements the same way. Instead, we use the
        # indices to sort the arrays, then compare the sorted masked arrays.
        # (The difference is that we don't compare the masked values.)
        i_sorted = np.asarray(res.data)
        res_data = np.take_along_axis(marrays[0].data, i_sorted, axis=-1)
        res_mask = np.take_along_axis(marrays[0].mask, i_sorted, axis=-1)
        res = mxp.asarray(res_data, mask=res_mask)
        ref_data = np.take_along_axis(masked_arrays[0].data, ref, axis=-1)
        ref_mask = np.take_along_axis(masked_arrays[0].mask, ref, axis=-1)
        ref = np.ma.MaskedArray(ref_data, mask=ref_mask)
        assert_equal(res, ref, seed)


def test_import(xp=np):
    mxp = marray.masked_array(xp)
    from mxp import asarray
    asarray(10, mask=True)


def test_test():
    seed = 149020664425889521094089537542803361848
    # test_statistical_array('argmin', True, seed=seed)
    test_rarithmetic_binary(arithmetic_binary[0], 'float32', seed=seed)
