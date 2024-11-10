import pytest
import numpy as np
import marray

def get_arrays(n_arrays, xp=np, seed=None):
    xpm = marray.masked_array(xp)

    entropy = np.random.SeedSequence(seed).entropy
    rng = np.random.default_rng(entropy)

    ndim = rng.integers(1, 4)
    shape = rng.integers(1, 20, size=ndim)

    arrays = []
    masks = []
    for i in range(n_arrays):
        shape_mask = rng.random(size=ndim) > 0.85
        shape_i = shape.copy()
        shape_i[shape_mask] = 1
        arrays.append(rng.standard_normal(size=shape_i))
        # for now, make masks same shape as array
        # consider making them broadcastable to array shape
        # or broadcastable to the same shape
        masks.append(rng.random(size=shape_i) > 0.75)

    marrays = []
    masked_arrays = []
    for array, mask in zip(arrays, masks):
        marrays.append(xpm.asarray(array, mask=mask))
        masked_arrays.append(np.ma.masked_array(array, mask=mask))

    return marrays, masked_arrays, entropy

def assert_equal(res, ref, seed):
    try:
        np.testing.assert_equal(res.data[~res.mask], ref.data[~ref.mask])
        np.testing.assert_equal(res.mask, ref.mask)
    except AssertionError as e:
        raise AssertionError(seed) from e

unary_ops = [lambda x: +x, lambda x: -x]
binary_ops = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y,
              lambda x, y: x / y, lambda x, y: x // y, lambda x, y: x % y,
              lambda x, y: x ** y]

@pytest.mark.parametrize("op", unary_ops)
def test_unary(op, seed=None):
    marrays, masked_arrays, seed = get_arrays(1, seed=seed)
    res = op(marrays[0])
    ref = op(masked_arrays[0])
    assert_equal(res, ref, seed)

@pytest.mark.parametrize("op", binary_ops)
def test_binary(op, seed=None):
    marrays, masked_arrays, seed = get_arrays(2, seed=seed)
    res = op(marrays[0], marrays[1])
    ref = op(masked_arrays[0], masked_arrays[1])
    assert_equal(res, ref, seed)

def test_test():
    seed = 61942948054137721690525248999235107195
    test_binary(binary_ops[6], seed=seed)
