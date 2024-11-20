import math
import re
from functools import partial
import textwrap
import numpy as np

# Matches between a comma or closing bracket and a space or newline
replace_re = re.compile(r"(?<=['\]])(?=\s|$)")


def dedent(text):
    return textwrap.dedent(text.lstrip("\n").rstrip())


def indent_by_one(text):
    return textwrap.indent(text, " ").lstrip().rstrip()

def format_item(item, masked):
    return str(item) if not masked else "--"

def format_array_1d(data, mask, max_items=20, visible=3):
    if len(data) <= max_items or visible * 2 >= max_items:
        result = ", ".join(
            format_item(item, masked)
            for item, masked in zip(data, mask)
        )
    else:
        head = [
            format_item(item, masked)
            for item, masked in zip(data[:visible], mask[:visible])
        ]
        tail = [
            format_item(item, masked)
            for item, masked in zip(data[-visible:], mask[-visible:])
        ]

        result = ", ".join(head + ["..."] + tail)

    return f"[{result}]"


def format_array(data, mask, max_outer_items=10, max_inner_items=20, visible=3):
    if data.ndim == 0:
        return format_item(data, mask)
    elif data.ndim == 1:
        return format_array_1d(data, mask, max_items=max_inner_items, visible=visible)

    if data.ndim == 1:
        formatter = partial(
            format_array_1d,
            max_items=max_inner_items,
            visible=visible,
        )
    else:
        formatter = partial(
            format_array,
            max_outer_items=max_outer_items,
            max_inner_items=max_inner_items,
            visible=visible,
        )

    if data.shape[0] <= max_outer_items or visible * 2 >= max_outer_items:
        result = ",\n".join(
            formatter(data[index, ...], mask[index, ...])
            for index in range(data.shape[0])
        )
    else:
        head = [
            formatter(data[index, ...], mask[index, ...])
            for index in range(visible)
        ]
        tail = [
            formatter(data[index, ...], mask[index, ...])
            for index in range(-visible, 0)
        ]

        result = ",\n".join(head + ["..."] + tail)

    return f"[{indent_by_one(result)}]"


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
