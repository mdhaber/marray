import operator
import itertools

import numpy as np
import pytest
import array_api_strict as strict
import marray

dtypes_boolean = ['bool']
dtypes_integral = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64']
dtypes_real = ['float32', 'float64']
dtypes_complex = ['complex64', 'complex128']
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
            data = data.astype(dtype)

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


def assert_comparison(res, ref, seed, comparison, **kwargs):
    ref_mask = np.broadcast_to(ref.mask, ref.data.shape)
    try:
        comparison(res.data[~res.mask], ref.data[~ref_mask], **kwargs)
        comparison(res.mask, ref_mask, **kwargs)
    except AssertionError as e:
        raise AssertionError(seed) from e


def assert_equal(res, ref, seed, **kwargs):
    return assert_comparison(res, ref, seed, np.testing.assert_equal, **kwargs)


def assert_allclose(res, ref, seed, **kwargs):
    return assert_comparison(res, ref, seed, np.testing.assert_allclose, **kwargs)


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

# __array_namespace__
# __bool__
# __complex__
# __dlpack__
# __dlpack_device__
# __float__
# __getitem__
# __index__
# __int__
# __setitem__
# __to_device__

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
searching_array = ['argmax', 'argmin']  # NumPy masked array funcs not good references
statistical_array = ['cumulative_sum', 'max', 'mean',
                     'min', 'prod', 'std', 'sum', 'var']
utility_array = ['all', 'any']


@pytest.mark.parametrize("f", arithmetic_unary + arithmetic_methods_unary)
@pytest.mark.parametrize('dtype', dtypes_real)
def test_arithmetic_unary(f, dtype, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, seed=seed)
    res = f(marrays[0])
    ref = f(masked_arrays[0])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("f", arithmetic_binary + arithmetic_methods_binary)
def test_arithmetic_binary(f, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, seed=seed)
    res = f(marrays[0], marrays[1])
    ref_data = f(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = masked_arrays[0].mask | masked_arrays[1].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("f", array_binary + array_methods_binary)
def test_array_binary(f, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, seed=seed)
    if marrays[0].ndim < 2:
        with pytest.raises(ValueError, match="undefined"):
            f(marrays[0], marrays[0].mT)
    else:
        res = f(marrays[0], marrays[0].mT)

        x = masked_arrays[0].data
        mask = masked_arrays[0].mask
        x[mask] = 0
        data = f(x, x.mT)
        mask = ~f(~mask, ~mask.mT)
        ref = np.ma.masked_array(data, mask=mask)
        assert_allclose(res, ref, seed)


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
@pytest.mark.parametrize("f_name", statistical_array + utility_array)
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
    ref = np.ma.masked_array(ref.data, getattr(ref, 'mask', False))
    assert_equal(res, ref, seed)

# Test Linear Algebra functions

# Use Array API tests to test the following:
# Creation Functions (same behavior but with all-False mask)
# Data Type Functions (only `astype` remains to be tested)
# Elementwise function `clip` (all others are tested above)
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

#?
# Searching functions - would test argmin/argmax with statistical functions,
#                       but NumPy masked version isn't correct
# Set functions
# Sorting functions
# __array_namespace__

def test_test():
    seed = 149020664425889521094089537542803361848
    # test_statistical_array('argmin', True, seed=seed)
    test_rarithmetic_binary(arithmetic_binary[0], 'float32', seed=seed)
