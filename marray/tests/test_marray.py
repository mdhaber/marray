# TODO:
# - debug test_inplace failures
# - debug reciprocal operator failures
# - debug statistical function failures

import numpy as np
import pytest

import marray

dtypes_boolean = ['bool']
dtypes_integral = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64']
dtypes_real = ['float32', 'float64']
dtypes_complex = ['complex64', 'complex128']


def get_arrays(n_arrays, *, dtype='float64', xp=np, seed=None):
    xpm = marray.masked_array(xp)

    entropy = np.random.SeedSequence(seed).entropy
    rng = np.random.default_rng(entropy)

    ndim = rng.integers(1, 4)
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

def assert_equal(res, ref, seed):
    ref_mask = np.broadcast_to(ref.mask, ref.data.shape)
    try:
        np.testing.assert_equal(res.data[~res.mask], ref.data[~ref_mask])
        np.testing.assert_equal(res.mask, ref_mask)
    except AssertionError as e:
        raise AssertionError(seed) from e

arithmetic_unary = [lambda x: +x, lambda x: -x, abs]
arithmetic_binary = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y,
                     lambda x, y: x / y, lambda x, y: x // y, lambda x, y: x % y,
                     lambda x, y: x ** y]
# array operators
bitwise_unary = [lambda x: ~x]
bitwise_binary = [lambda x, y: x & y, lambda x, y: x | y, lambda x, y: x ^ y,
                  lambda x, y: x << y, lambda x, y: x >> y]
comparison_binary = [lambda x, y: x < y, lambda x, y: x <= y, lambda x, y: x > y,
                     lambda x, y: x >= y, lambda x, y: x == y , lambda x, y: x != y]

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


elementwise_unary = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
                     'ceil', 'cos', 'cosh', 'exp', 'expm1',
                     'floor', 'isfinite', 'isinf', 'isnan', 'log', 'log1p',
                     'log2', 'log10', 'logical_not', 'negative', 'positive',
                     'sign', 'signbit', 'sin', 'sinh', 'square', 'sqrt', 'tan',
                     'tanh', 'trunc']
elementwise_binary = ['add', 'atan2', 'copysign', 'divide', 'equal', 'floor_divide',
                      'greater', 'greater_equal', 'hypot', 'less', 'less_equal',
                      'logaddexp', 'logical_and', 'logical_or', 'logical_xor',
                      'maximum', 'minimum', 'multiply', 'not_equal', 'pow', 'subtract']
statistical_array = ['cumulative_sum', 'max', 'mean',
                     'min', 'prod', 'std', 'sum', 'var']

"""
'bitwise_invert'
'bitwise_and'
'bitwise_left_shift'
'bitwise_or'
'bitwise_right_shift'
'bitwise_xor'

'logical_and'
'logical_not'

'clip'
'conj'
'real'
"""

@pytest.mark.parametrize("f", arithmetic_unary)
def test_arithmetic_unary(f, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, seed=seed)
    res = f(marrays[0])
    ref = f(masked_arrays[0])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("f", arithmetic_binary)
def test_arithmetic_binary(f, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, seed=seed)
    res = f(marrays[0], marrays[1])
    ref_data = f(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = masked_arrays[0].mask | masked_arrays[1].mask
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize("f", bitwise_unary)
def test_bitwise_unary(f, dtype, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, seed=seed)
    res = f(~marrays[0])
    ref = f(~masked_arrays[0])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize("f", bitwise_binary)
def test_bitwise_binary(f, dtype, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
    res = f(marrays[0], marrays[1])
    ref = f(masked_arrays[0], masked_arrays[1])
    assert_equal(res, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_real)
@pytest.mark.parametrize("f", comparison_binary)
def test_comparison_binary(f, dtype, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
    res = f(marrays[0], marrays[1])
    ref = f(masked_arrays[0], masked_arrays[1])
    assert_equal(res, ref, seed)


# @pytest.mark.parametrize("dtype", dtypes_integral + dtypes_real)
# @pytest.mark.parametrize("f", inplace_arithmetic + inplace_bitwise)
# def test_inplace(f, dtype, seed=None):
#     marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)
#     e1 = None
#     e2 = None
#
#     try:
#         f(masked_arrays[0], masked_arrays[1])
#     except Exception as e:
#         e1 = e
#     try:
#         f(marrays[0], marrays[1])
#     except Exception as e:
#         e2 = e
#
#     if e1 or e2:
#         assert str(e1) == str(e2)
#     else:
#         assert_equal(marrays[0], masked_arrays[0], seed)


@pytest.mark.parametrize("f", arithmetic_binary)
def test_rarithmetic_binary(f, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, seed=seed)

    res = f(marrays[0], marrays[1].data)
    ref_data = f(masked_arrays[0].data, masked_arrays[1].data)
    ref_mask = np.broadcast_to(masked_arrays[0].mask, ref_data.shape)
    ref = np.ma.masked_array(ref_data, mask=ref_mask)
    assert_equal(res, ref, seed)

    # res = f(marrays[1].data, marrays[0])
    # ref = f(masked_arrays[1].data, masked_arrays[0])
    # assert_equal(res, ref, seed)


@pytest.mark.parametrize("dtype", dtypes_integral + dtypes_boolean)
@pytest.mark.parametrize("f", bitwise_binary)
def test_rbitwise_binary(f, dtype, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, dtype=dtype, seed=seed)

    res = f(marrays[0], marrays[1].data)
    ref = f(masked_arrays[0], masked_arrays[1].data)
    assert_equal(res, ref, seed)

    # res = f(marrays[1].data, marrays[0])
    # ref = f(masked_arrays[1].data, masked_arrays[0])
    # assert_equal(res, ref, seed)


def test_constants(xp=np):
    mxp = marray.masked_array(xp)
    assert mxp.e == xp.e
    assert mxp.inf == xp.inf
    assert np.isnan(mxp.nan) == np.isnan(xp.nan)
    assert mxp.newaxis == xp.newaxis
    assert mxp.pi == xp.pi


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


# @pytest.mark.parametrize("keepdims", [False, True])
# @pytest.mark.parametrize("f_name", statistical_array)
# def test_statistical_array(f_name, keepdims, xp=np, dtype='float64', seed=None):
#     # TODO: confirm that result should never have mask? Only when all are masked?
#     mxp = marray.masked_array(xp)
#     marrays, masked_arrays, seed = get_arrays(1, dtype=dtype, seed=seed)
#     rng = np.random.default_rng(seed)
#     axes = list(range(marrays[0].ndim)) + [None]
#     axis = axes[rng.integers(marrays[0].ndim + 1)]
#     f = getattr(mxp, f_name)
#     f2 = getattr(xp, f_name)
#     kwargs = {'keepdims': keepdims} if f_name != 'cumulative_sum' else {}
#     res = f(marrays[0], axis=axis, **kwargs)
#     ref = f2(masked_arrays[0], axis=axis, **kwargs)
#     assert_equal(res, ref, seed)


def test_test():
    seed = 8377009968503871097350278305436713931
    test_rarithmetic_binary(arithmetic_binary[0], seed=seed)
