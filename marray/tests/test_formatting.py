import pytest
import numpy as np
import marray
from marray import formatting


xp = marray.masked_array(np)

def construct_marray(data, mask):
    arr = xp.asarray(data)
    arr.mask[...] = mask

    return arr


@pytest.mark.parametrize(["arr", "expected"], (
    pytest.param(construct_marray(np.array(0), True), "--", id="0d-masked"),
    pytest.param(construct_marray(np.array(0), False), "0", id="0d-unmasked"),
    pytest.param(construct_marray(np.arange(2), [False, True]), "[0, --]", id="1d"),
    pytest.param(
        construct_marray(np.arange(6).reshape(2, 3), [[False, True, False], [True, False, True]]),
        "[[0, --, 2],\n [--, 4, --]]",
        id="2d",
    ),
    pytest.param(
        construct_marray(
            np.arange(12).reshape(2, 3, 2),
            [[[False, True], [True, False], [False, True]], [[True, False], [False, True], [True, False]]]
        ),
        "[[[0, --],\n  [--, 3],\n  [4, --]],\n\n [[--, 7],\n  [8, --],\n  [--, 11]]]",
        id="3d",
    ),
))
def test_format_data(arr, expected):
    actual = formatting.format_data(arr.data, arr.mask)

    assert actual == expected
