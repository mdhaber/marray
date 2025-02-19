---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tutorial

MArray is a package for extending your favorite [Python Array API Standard](https://data-apis.org/array-api/latest/index.html) compatible library with mask capabilities. Motivation for masked arrays can be found at ["What is a masked array?"](https://numpy.org/devdocs/reference/maskedarray.generic.html#what-is-a-masked-array).

MArray is easy to install with `pip`, and it has no required dependencies.

```{code-cell} ipython3
# !pip install marray
```

The rest of the tutorial will assume that we want to add masks to NumPy arrays. Note that this is different from using NumPy's built-in masked arrays from the `numpy.ma` namespace because `numpy.ma` is not compatible with the array API standard. Even the base NumPy namespace is not Array API compatible in versions of NumPy prior to 2.0, so we will install a recent version of NumPy to work with.

```{code-cell} ipython3
# !pip install --upgrade numpy
```

To create a version of the NumPy namespace with mask support, use Python's `from...import...as` syntax.

```{code-cell} ipython3
from marray import numpy as mxp
```

For cases in which this syntax would fail, use `marray`'s only public function: `marray.masked_namespace`.

```{code-cell} ipython3
from array_api_compat import numpy as np
import marray
mxp = marray.masked_namespace(np)
```

`mxp` exposes all the features of NumPy that are specified in the Array API standard, but adds masks support to them. For example:

```{code-cell} ipython3
x = mxp.arange(3)
x
```

Just as `xp.arange(3)` would have created a regular NumPy array with elements [0, 1, 2], `mxp.arange(3)` creates an `MArray` object with these elements. These are accessible via the `data` attribute.

```{code-cell} ipython3
x.data
```

The difference is that the `MArray` also has a mask, available via the `mask` attribute.

```{code-cell} ipython3
x.mask
```

Because all of the elements of the mask are `False`, this `MArray` will behave just like a regular NumPy array. That's boring. Let's create an array with a nontrivial mask. To do that, we'll use `mxp.asarray`.

```{code-cell} ipython3
x = mxp.asarray([1, 2, 3, 4], mask=[False, True, False, True])
x
```

`marray` is intended to be a very light wrapper of the underlying array library. Just as it has only one public function (`get_namespace`), it makes only one modification to the signature of a wrapped library function, which we've used above: it adds a `mask` keyword-only argument to the `asarray` function.

Let's see how the mask changes the behavior of common functions.

## Statistical Functions
For reducing functions, masked elements are ignored; the result is the same as if the masked elements were not in the array.

```{code-cell} ipython3
mxp.max(x)  # 4 was masked
```

```{code-cell} ipython3
mxp.sum(x)  # 2 and 4 were masked
```

For the only non-reducing statistical function, `cumulative_sum`, masked elements do not contribute to the cumulative sum.

```{code-cell} ipython3
mxp.cumulative_sum(x)
```

Note that the elements at indices where the original array were masked remain masked. Because of the limitations of the underlying array library, there will always be values corresponding with masked elements in `data`, *but these values should be considered meaningless*.

## Utility functions
`all` and `any` work like the reducing statistics functions.

```{code-cell} ipython3
x = mxp.asarray([False, False, False, True], mask=[False, True, False, True])
mxp.all(x)
```

```{code-cell} ipython3
mxp.any(x)
```

Is that last result surprising? Although there is one `True` in `x.data`, it is ignored when computing `any` because it is masked.

You may have noticed that the mask of the result has always been `False` in these examples of reducing functions. This is always the case unless *all* elements of the array are masked. In this case, it is required by the reducing nature of the function to return a 0D array for a 1D input, but there is not an universally accepted result for these functions when all elements are masked. (What is the maximum of an empty set?)

```{code-cell} ipython3
x = mxp.asarray(x.data, mask=True)
mxp.any(x).mask
```

## Sorting functions
The sorting functions treat masked values as undefined and, by convention, append them to the end of the returned array.

```{code-cell} ipython3
data = [8, 3, 4, 1, 9, 9, 5, 5]
mask = [0, 0, 1, 0, 1, 1, 0, 0]
x = mxp.asarray(data, mask=mask)
mxp.sort(x)
```

Where did those huge numbers come from? We emphasize again: *the `data` corresponding with masked elements should be considered meaningless*; they are just placeholders that allow us respect the mask while doing array operations efficiently.

```{code-cell} ipython3
i = mxp.argsort(x)
i
```

Is it surprising that the mask of the array returned by `argsort` is all False? These are the indices that allow us to transform the original array into the sorted result. We can confirm that without a mask, these indices sort the array and keep the right elements masked.

```{code-cell} ipython3
y = x[i.data]
y
```

*Gotcha:* Sorting is not supported when the the non-masked data includes the maximum (minimum when `descending=True`) value of the data's `dtype`.

```{code-cell} ipython3
z = mxp.asarray(x, mask=mask, dtype=mxp.uint8)
z[0] = 2**8 - 1
# mxp.sort(z)
# NotImplementedError: The maximum value of the data's dtype is included in the non-masked data; this complicates sorting when masked values are present.
# Consider promoting to another dtype to use `sort`.
```

It is often possible to sidestep this limitation by using a different `dtype` for the sorting, then converting back to the original type.

```{code-cell} ipython3
z = mxp.astype(z, mxp.uint16)
z_sorted = mxp.sort(z)
z_sorted = mxp.astype(z_sorted, mxp.uint8)
z_sorted
```

## Set functions
Masked elements are treated as distinct from all non-masked elements but equivalent to all other masked elements.

```{code-cell} ipython3
res = mxp.unique_counts(x)
```

```{code-cell} ipython3
res.values
```

```{code-cell} ipython3
res.counts
```

*Gotcha*: set functions have the same limitation as the sorting functions: the non-masked data may not include the maximum value of the data's `dtype`.

+++

## Manipulation functions
Manipulation functions perform the same operation on the data and the mask.

```{code-cell} ipython3
mxp.flip(y)
```

```{code-cell} ipython3
mxp.stack([y, y])
```

## Creation functions
Most creation functions create arrays with an all-False mask.

```{code-cell} ipython3
mxp.eye(3)
```

Exceptions include the `_like` functions, which preserve the mask of the array argument.

```{code-cell} ipython3
mxp.zeros_like(y)
```

`tril` and `triu` also preserve the mask of the indicated triangular portion of the argument.

```{code-cell} ipython3
import numpy as xp

data = xp.ones((3, 3))
mask = xp.zeros_like(data)
mask[0, -1] = 1
mask[-1, 0] = 1
A = mxp.asarray(data, mask=mask)
A
```

```{code-cell} ipython3
mxp.tril(A)
```

## Searching functions
Similarly to the statistics functions, masked elements are treated as if they did not exist.

```{code-cell} ipython3
x[[1, -1]] = 0  # add some zeros
x  # let's remember what `x` looks like
```

```{code-cell} ipython3
mxp.argmax(x)  # 9 is masked, so 8 (at index 0) is the largest element
```

```{code-cell} ipython3
i = mxp.nonzero(x)  # Only elements at these indices are nonzero *and* not masked
i
```

The correct behavior of indexing with a masked array is ambiguous, so use only regular, unmasked arrays for indexing.

```{code-cell} ipython3
indices = i[0].data
x[indices]  # nonzero, not masked
```

## Elementwise functions
Elementwise functions (and operators) simply perform the requested operation on the `data`.

For unary functions, the mask of the result is the mask of the argument.

```{code-cell} ipython3
x = xp.linspace(0, 2*xp.pi, 5)
x = mxp.asarray(x, mask=(x > xp.pi))
x
```

```{code-cell} ipython3
-x
```

```{code-cell} ipython3
mxp.round(mxp.sin(x))
```

For binary functions and operators, the mask of the result is the result of the logical *or* operation on the masks of the arguments.

```{code-cell} ipython3
x = mxp.asarray([1, 2, 3, 4], mask=[1, 0, 1, 0])
y = mxp.asarray([5, 6, 7, 8], mask=[1, 1, 0, 0])
x + y
```

```{code-cell} ipython3
mxp.pow(y, x)
```

Note that `np.ma` automatically masks non-finite elements produced during calculations.

```{code-cell} ipython3
import numpy

x = numpy.ma.masked_array(0, mask=False)
with numpy.errstate(divide='ignore', invalid='ignore'):
    y = [1, 0] / x
y
```

`MArray` *does not* follow this convention.

```{code-cell} ipython3
x = mxp.asarray(0, mask=False)
with numpy.errstate(divide='ignore', invalid='ignore'):
    y = [1, 0] / x
y
```

This is because masked elements are often used to represent *missing* data, and the results of these operations are not missing. If this does not suit your needs, mask out data according to your requirements after performing the operation.

```{code-cell} ipython3
x = mxp.asarray(0, mask=False)
with numpy.errstate(divide='ignore', invalid='ignore'):
    y = [1, 0] / x
mxp.asarray(y.data, mask=xp.isnan(y.data))
```

## Linear Algebra Functions
As usual, linear algebra functions and operators treat masked elements as though they don't exist.

```{code-cell} ipython3
x = mxp.asarray([1, 2, 3, 4], mask=[1, 0, 1, 0])
y = mxp.asarray([5, 6, 7, 8], mask=[1, 1, 0, 0])
x @ y  # the last elements of the arrays, 4 and 8, are the only non-masked elements
```

The exception is `matrix_transpose`, which transposes the data and the mask.

```{code-cell} ipython3
x = mxp.asarray([[1, 2], [3, 4]], mask=[[1, 1], [0, 0]])
x
```

```{code-cell} ipython3
mxp.matrix_transpose(x)
```

## Conclusion
While this tutorial is not exhaustive, we hope it is sufficient to allow you to predict the results of operations with `MArray`s and use them to suit your needs. If you'd like to see this tutorial extended in a particular way, please [open an issue](https://github.com/mdhaber/marray/issues)!
