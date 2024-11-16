import re
from functools import partial
import textwrap
import numpy as np

# Matches between a comma or closing bracket and a space or newline
replace_re = re.compile(r"(?<=['\]])(?=\s|$)")


def as_masked_array(arr):
    # temporary: fix for CuPy
    # eventually: rewrite to avoid masked array
    data = np.asarray(arr.data)
    mask = np.asarray(arr.mask)
    return np.ma.masked_array(data, mask)


def dedent(text):
    return textwrap.dedent(text.lstrip("\n").rstrip())


def format_data(data, mask):
    # Format data by converting to a string dtype, masking out the masked values,
    # then relying on numpy.ndarray's repr
    # TODO: This won't work for string dtypes, so to support those we need to write
    # our own array formatting
    try:
        dtype = np.dtypes.StringDType
    except AttributeError:
        dtype = "str"

    formatted = np.where(mask, "--", data.astype(dtype))
    with_commas = replace_re.sub(",", str(formatted)).rstrip(",")
    return with_commas.replace("'", "")


def format_repr(arr):
    data_repr = format_data(arr.data, arr.mask)

    template = dedent(
        """
        <marray.MaskedArray>
        Data:
        {data_repr}

        Mask:
        {mask_repr}

        fill_value: {fill_value}
        """
    )

    return template.format(
        data_repr=data_repr,
        mask_repr=str(arr.mask),
        fill_value=str(arr._sentinel)
    )
