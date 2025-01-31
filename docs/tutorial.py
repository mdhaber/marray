import marimo

__generated_with = "0.10.18"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Tutorial

        MArray is a package for extending your favorite [Python Array API Standard](https://data-apis.org/array-api/latest/index.html) compatible library with mask capabilities. Motivation for masked arrays can be found at ["What is a masked array?"](https://numpy.org/devdocs/reference/maskedarray.generic.html#what-is-a-masked-array).

        MArray is easy to install with `pip`, and it has no required dependencies.

        The rest of the tutorial will assume that we want to add masks to NumPy arrays. Note that this is different from using NumPy's built-in masked arrays from the `numpy.ma` namespace because `numpy.ma` is not compatible with the array API standard. Even the base NumPy namespace is not Array API compatible in versions of NumPy prior to 2.0, so we will install a recent version of NumPy to work with.

        To create a version of the NumPy namespace with mask support, use Python's `from...import...as` syntax.
        """
    )
    return


@app.cell
def _():
    from marray import numpy as mxp
    return (mxp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""`mxp` exposes all the features of NumPy that are specified in the Array API standard, but adds masks support to them. For example:""")
    return


@app.cell
def _(mxp):
    simple_array = mxp.arange(3)
    repr(simple_array)
    return (simple_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Just as `xp.arange(3)` would have created a regular NumPy array with elements [0, 1, 2], `mxp.arange(3)` creates an `MArray` object with these elements. These are accessible via the `data` attribute.""")
    return


@app.cell
def _(simple_array):
    repr(simple_array.data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The difference is that the `MArray` also has a mask, available via the `mask` attribute.""")
    return


@app.cell
def _(simple_array):
    repr(simple_array.mask)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Because all of the elements of the mask are `False`, this `MArray` will behave just like a regular NumPy array. That's boring. Let's create an array with a nontrivial mask. To do that, we'll use `mxp.asarray`.""")
    return


@app.cell
def _(mxp):
    x = mxp.asarray([1, 2, 3, 4], mask=[False, True, False, True])
    repr(x)
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        `marray` is intended to be a very light wrapper of the underlying array library. Just as it has only one public function (`get_namespace`), it makes only one modification to the signature of a wrapped library function, which we've used above: it adds a `mask` keyword-only argument to the `asarray` function.

        Let's see how the mask changes the behavior of common functions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Statistical Functions
        For reducing functions, masked elements are ignored; the result is the same as if the masked elements were not in the array.
        """
    )
    return


@app.cell
def _(mxp, x):
    repr(mxp.max(x))  # 4 was masked
    return


@app.cell
def _(mxp, x):
    repr(mxp.sum(x))  # 2 and 4 were masked
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""For the only non-reducing statistical function, `cumulative_sum`, masked elements do not contribute to the cumulative sum.""")
    return


@app.cell
def _(mxp, x):
    repr(mxp.cumulative_sum(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Note that the elements at indices where the original array were masked remain masked. Because of the limitations of the underlying array library, there will always be values corresponding with masked elements in `data`, *but these values should be considered meaningless*.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Utility functions
        `all` and `any` work like the reducing statistics functions.
        """
    )
    return


@app.cell
def _(mxp):
    y = mxp.asarray([False, False, False, True], mask=[False, True, False, True])
    repr(mxp.all(y))
    return (y,)


@app.cell
def _(mxp, y):
    repr(mxp.any(y))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Is that last result surprising? Although there is one `True` in `x.data`, it is ignored when computing `any` because it is masked.

        You may have noticed that the mask of the result has always been `False` in these examples of reducing functions. This is always the case unless *all* elements of the array are masked. In this case, it is required by the reducing nature of the function to return a 0D array for a 1D input, but there is not an universally accepted result for these functions when all elements are masked. (What is the maximum of an empty set?)
        """
    )
    return


@app.cell
def _(mxp, y):
    y_all_masked = mxp.asarray(y.data, mask=True)
    repr(mxp.any(y_all_masked).mask)
    return (y_all_masked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Sorting functions
        The sorting functions treat masked values as undefined and, by convention, append them to the end of the returned array.
        """
    )
    return


@app.cell
def _(mxp):
    z_data = [8, 3, 4, 1, 9, 9, 5, 5]
    z_mask = [0, 0, 1, 0, 1, 1, 0, 0]
    z = mxp.asarray(z_data, mask=z_mask)
    repr(mxp.sort(z))
    return z, z_data, z_mask


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Where did those huge numbers come from? We emphasize again: *the `data` corresponding with masked elements should be considered meaningless*; they are just placeholders that allow us respect the mask while doing array operations efficiently.""")
    return


@app.cell
def _(mxp, z):
    i = mxp.argsort(z)
    repr(i)
    return (i,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Is it surprising that the mask of the array returned by `argsort` is all False? These are the indices that allow us to transform the original array into the sorted result. We can confirm that without a mask, these indices sort the array and keep the right elements masked.""")
    return


@app.cell
def _(i, z):
    a = z[i.data]
    repr(a)
    return (a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""*Gotcha:* Sorting is not supported when the the non-masked data includes the maximum (minimum when `descending=True`) value of the data's `dtype`.""")
    return


@app.cell
def _(mxp, z, z_mask):
    b = mxp.asarray(z, mask=z_mask, dtype=mxp.uint8)
    b[0] = 2**8 - 1
    # mxp.sort(b)
    # NotImplementedError: The maximum value of the data's dtype is included in the non-masked data; this complicates sorting when masked values are present.
    # Consider promoting to another dtype to use `sort`.
    return (b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""It is often possible to sidestep this limitation by using a different `dtype` for the sorting, then converting back to the original type.""")
    return


@app.cell
def _(b, mxp):
    c = mxp.astype(b, mxp.uint16)
    c_sorted = mxp.astype(mxp.sort(c), mxp.uint8)
    repr(c_sorted)
    return c, c_sorted


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Set functions
        Masked elements are treated as distinct from all non-masked elements but equivalent to all other masked elements.
        """
    )
    return


@app.cell
def _(mxp, z):
    z_unique = mxp.unique_counts(z)
    repr(z_unique.values)
    return (z_unique,)


@app.cell
def _(z_unique):
    repr(z_unique.counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""*Gotcha*: set functions have the same limitation as the sorting functions: the non-masked data may not include the maximum value of the data's `dtype`.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Manipulation functions
        Manipulation functions perform the same operation on the data and the mask.
        """
    )
    return


@app.cell
def _(a, mxp):
    repr(mxp.flip(a))
    return


@app.cell
def _(a, mxp):
    repr(mxp.stack([a, a]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Creation functions
        Most creation functions create arrays with an all-False mask.
        """
    )
    return


@app.cell
def _(mxp):
    repr(mxp.eye(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Exceptions include the `_like` functions, which preserve the mask of the array argument.""")
    return


@app.cell
def _(a, mxp):
    repr(mxp.zeros_like(a))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""`tril` and `triu` also preserve the mask of the indicated triangular portion of the argument.""")
    return


@app.cell
def _(mxp):
    import numpy as xp

    A_data = xp.ones((3, 3))
    A_mask = xp.zeros_like(A_data)
    A_mask[0, -1] = 1
    A_mask[-1, 0] = 1
    A = mxp.asarray(A_data, mask=A_mask)
    repr(A)
    return A, A_data, A_mask, xp


@app.cell
def _(A, mxp):
    repr(mxp.tril(A))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Searching functions
        Similarly to the statistics functions, masked elements are treated as if they did not exist.
        """
    )
    return


@app.cell
def _(z):
    z_with_zeros = z
    z_with_zeros[[1, -1]] = 0  # add some zeros
    repr(z_with_zeros)  # let's remember what `z` looks like
    return (z_with_zeros,)


@app.cell
def _(mxp, z_with_zeros):
    repr(mxp.argmax(z_with_zeros))  # 9 is masked, so 8 (at index 0) is the largest element
    return


@app.cell
def _(mxp, z_with_zeros):
    z_i = mxp.nonzero(z_with_zeros)  # Only elements at these indices are nonzero *and* not masked
    repr(z_i)
    return (z_i,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The correct behavior of indexing with a masked array is ambiguous, so use only regular, unmasked arrays for indexing.""")
    return


@app.cell
def _(z_i, z_with_zeros):
    indices = z_i[0].data
    repr(z_with_zeros[indices])
    return (indices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Elementwise functions
        Elementwise functions (and operators) simply perform the requested operation on the `data`.

        For unary functions, the mask of the result is the mask of the argument.
        """
    )
    return


@app.cell
def _(mxp, xp):
    d = xp.linspace(0, 2*xp.pi, 5)
    d = mxp.asarray(d, mask=(d > xp.pi))
    repr(d)
    return (d,)


@app.cell
def _(d):
    repr(-d)
    return


@app.cell
def _(d, mxp):
    repr(mxp.round(mxp.sin(d)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""For binary functions and operators, the mask of the result is the result of the logical *or* operation on the masks of the arguments.""")
    return


@app.cell
def _(mxp):
    e = mxp.asarray([1, 2, 3, 4], mask=[1, 0, 1, 0])
    f = mxp.asarray([5, 6, 7, 8], mask=[1, 1, 0, 0])
    repr(e + f)
    return e, f


@app.cell
def _(e, f, mxp):
    repr(mxp.pow(f, e))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Note that `np.ma` automatically masks non-finite elements produced during calculations.""")
    return


@app.cell
def _():
    import numpy

    g = numpy.ma.masked_array(0, mask=False)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        h = [1, 0] / g
    repr(h)
    return g, h, numpy


@app.cell(hide_code=True)
def _(mo):
    mo.md("""`MArray` *does not* follow this convention.""")
    return


@app.cell
def _(mxp, numpy):
    j = mxp.asarray(0, mask=False)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        k = [1, 0] / j
    repr(k)
    return j, k


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This is because masked elements are often used to represent *missing* data, and the results of these operations are not missing. If this does not suit your needs, mask out data according to your requirements after performing the operation.""")
    return


@app.cell
def _(mxp, numpy, xp):
    m = mxp.asarray(0, mask=False)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        n = [1, 0] / m
    repr(mxp.asarray(n.data, mask=xp.isnan(n.data)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Linear Algebra Functions
        As usual, linear algebra functions and operators treat masked elements as though they don't exist.
        """
    )
    return


@app.cell
def _(mxp):
    o = mxp.asarray([1, 2, 3, 4], mask=[1, 0, 1, 0])
    p = mxp.asarray([5, 6, 7, 8], mask=[1, 1, 0, 0])
    repr(o @ p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The exception is `matrix_transpose`, which transposes the data and the mask.""")
    return


@app.cell
def _(mxp):
    q = mxp.asarray([[1, 2], [3, 4]], mask=[[1, 1], [0, 0]])
    repr(q)
    return (q,)


@app.cell
def _(mxp, q):
    repr(mxp.matrix_transpose(q))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Conclusion
        While this tutorial is not exhaustive, we hope it is sufficient to allow you to predict the results of operations with `MArray`s and use them to suit your needs. If you'd like to see this tutorial extended in a particular way, please [open an issue](https://github.com/mdhaber/marray/issues)!
        """
    )
    return


if __name__ == "__main__":
    app.run()
