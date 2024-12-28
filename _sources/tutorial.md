### Tutorial

MArray is a package for extending your favorite [Python Array API Standard](https://data-apis.org/array-api/latest/index.html) compatible library with mask capabilities. Motivation for masked arrays can be found at ["What is a masked array?"](https://numpy.org/devdocs/reference/maskedarray.generic.html#what-is-a-masked-array).

MArray is easy to install with `pip`, and it has no required dependencies.


```python
!pip install marray
```

    Requirement already satisfied: marray in /usr/local/lib/python3.10/dist-packages (0.0.5)
    

The rest of the tutorial will assume that we want to add masks to NumPy arrays. Note that this is different from using NumPy's built-in masked arrays from the `numpy.ma` namespace because `numpy.ma` is not compatible with the array API standard. Even the base NumPy namespace is not Array API compatible in versions of NumPy prior to 2.0, so we will install a recent version of NumPy to work with.


```python
!pip install --upgrade numpy
```

    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (2.2.0)
    

To create a version of the NumPy namespace with mask support, use `marray`'s only public attribute: `get_namespace`.


```python
import numpy as xp

import marray

mxp = marray.get_namespace(xp)
```

`mxp` exposes all the features of NumPy that are specified in the Array API standard, but adds masks support to them. For example:


```python
x = mxp.arange(3)
x
```




    MArray(array([0, 1, 2]), array([False, False, False]))



Just as `xp.arange(3)` would have created a regular NumPy array with elements [0, 1, 2], `mxp.arange(3)` creates an `MArray` object with these elements. These are accessible via the `data` attribute.


```python
x.data
```




    array([0, 1, 2])



The difference is that the `MArray` also has a mask, available via the `mask` attribute.


```python
x.mask
```




    array([False, False, False])



Because all of the elements of the mask are `False`, this `MArray` will behave just like a regular NumPy array. That's boring. Let's create an array with a nontrivial mask. To do that, we'll use `mxp.asarray`.


```python
x = mxp.asarray([1, 2, 3, 4], mask=[False, True, False, True])
x
```




    MArray(array([1, 2, 3, 4]), array([False,  True, False,  True]))



`marray` is intended to be a very light wrapper of the underlying array library. Just as it has only one public function (`get_namespace`), it makes only one modification to the signature of a wrapped library function, which we've used above: it adds a `mask` keyword-only argument to the `asarray` function.

Let's see how the mask changes the behavior of common functions.

#### Statistical Functions
For reducing functions, masked elements are ignored; the result is the same as if the masked elements were not in the array.


```python
mxp.max(x)  # 4 was masked
```




    MArray(array(3), array(False))




```python
mxp.sum(x)  # 2 and 4 were masked
```




    MArray(array(4), array(False))



For the only non-reducing statistical function, `cumulative_sum`, masked elements do not contribute to the cumulative sum.


```python
mxp.cumulative_sum(x)
```




    MArray(array([1, 1, 4, 4]), array([False,  True, False,  True]))



Note that the elements at indices where the original array were masked remain masked. Because of the limitations of the underlying array library, there will always be values corresponding with masked elements in `data`, *but these values should be considered meaningless*.

#### Utility functions
`all` and `any` work like the reducing statistics functions.


```python
x = mxp.asarray([False, False, False, True], mask=[False, True, False, True])
mxp.all(x)
```




    MArray(array(False), array(False))




```python
mxp.any(x)
```




    MArray(array(False), array(False))



Is that last result surprising? Although there is one `True` in `x.data`, it is ignored when computing `any` because it is masked.

You may have noticed that the mask of the result has always been `False` in these examples of reducing functions. This is always the case unless *all* elements of the array are masked. In this case, it is required by the reducing nature of the function to return a 0D array for a 1D input, but there is not an universally accepted result for these functions when all elements are masked. (What is the maximum of an empty set?)


```python
x = mxp.asarray(x.data, mask=True)
mxp.any(x).mask
```




    array(True)



#### Sorting functions
The sorting functions treat masked values as undefined and, by convention, append them to the end of the returned array.


```python
data = [8, 3, 4, 1, 9, 9, 5, 5]
mask = [0, 0, 1, 0, 1, 1, 0, 0]
x = mxp.asarray(data, mask=mask)
mxp.sort(x)
```




    MArray(
        array([                  1,                   3,                   5,
                             5,                   8, 9223372036854775807,
           9223372036854775807, 9223372036854775807]),
        array([False, False, False, False, False,  True,  True,  True])
    )



Where did those huge numbers come from? We emphasize again: *the `data` corresponding with masked elements should be considered meaningless*; they are just placeholders that allow us respect the mask while doing array operations efficiently.


```python
i = mxp.argsort(x)
i
```




    MArray(
        array([3, 1, 6, 7, 0, 2, 4, 5]),
        array([False, False, False, False, False, False, False, False])
    )



Is it surprising that the mask of the array returned by `argsort` is all False? These are the indices that allow us to transform the original array into the sorted result. We can confirm that without a mask, these indices sort the array and keep the right elements masked.


```python
y = x[i.data]
y
```




    MArray(
        array([1, 3, 5, 5, 8, 4, 9, 9]),
        array([False, False, False, False, False,  True,  True,  True])
    )



#### Manipulation functions
Manipulation functions perform the same operation on the data and the mask.


```python
mxp.flip(y)
```




    MArray(
        array([9, 9, 4, 8, 5, 5, 3, 1]),
        array([ True,  True,  True, False, False, False, False, False])
    )




```python
mxp.stack([y, y])
```




    MArray(
        array([[1, 3, 5, 5, 8, 4, 9, 9],
           [1, 3, 5, 5, 8, 4, 9, 9]]),
        array([[False, False, False, False, False,  True,  True,  True],
           [False, False, False, False, False,  True,  True,  True]])
    )



#### Creation functions
Most creation functions create arrays with an all-False mask.


```python
mxp.eye(3)
```




    MArray(
        array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]),
        array([[False, False, False],
           [False, False, False],
           [False, False, False]])
    )



Exceptions include the `_like` functions, which preserve the mask of the array argument.


```python
mxp.zeros_like(y)
```




    MArray(
        array([0, 0, 0, 0, 0, 0, 0, 0]),
        array([False, False, False, False, False,  True,  True,  True])
    )



`tril` and `triu` also preserve the mask of the indicated triangular portion of the argument.


```python
data = xp.ones((3, 3))
mask = xp.zeros_like(data)
mask[0, -1] = 1
mask[-1, 0] = 1
A = mxp.asarray(data, mask=mask)
A
```




    MArray(
        array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]]),
        array([[False, False,  True],
           [False, False, False],
           [ True, False, False]])
    )




```python
mxp.tril(A)
```




    MArray(
        array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]]),
        array([[False, False, False],
           [False, False, False],
           [ True, False, False]])
    )



#### Searching functions
Similarly to the statistics functions, masked elements are treated as if they did not exist.


```python
x[[1, -1]] = 0  # add some zeros
x  # let's remember what `x` looks like
```




    MArray(
        array([8, 0, 4, 1, 9, 9, 5, 0]),
        array([False, False,  True, False,  True,  True, False, False])
    )




```python
mxp.argmax(x)  # 9 is masked, so 8 (at index 0) is the largest element
```




    MArray(array(0), array(False))




```python
i = mxp.nonzero(x)  # Only elements at these indices are nonzero *and* not masked
i
```




    (MArray(array([0, 3, 6]), array([False, False, False])),)



The correct behavior of indexing with a masked array is ambiguous, so use only regular, unmasked arrays for indexing.


```python
indices = i[0].data
x[indices]  # nonzero, not masked
```




    MArray(array([8, 1, 5]), array([False, False, False]))



#### Elementwise functions
Elementwise functions (and operators) simply perform the requested operation on the `data`.

For unary functions, the mask of the result is the mask of the argument.


```python
x = xp.linspace(0, 2*xp.pi, 5)
x = mxp.asarray(x, mask=(x > xp.pi))
x
```




    MArray(
        array([0.        , 1.57079633, 3.14159265, 4.71238898, 6.28318531]),
        array([False, False, False,  True,  True])
    )




```python
-x
```




    MArray(
        array([-0.        , -1.57079633, -3.14159265, -4.71238898, -6.28318531]),
        array([False, False, False,  True,  True])
    )




```python
mxp.round(mxp.sin(x))
```




    MArray(
        array([ 0.,  1.,  0., -1., -0.]),
        array([False, False, False,  True,  True])
    )



For binary functions and operators, the mask of the result is the result of the logical *or* operation on the masks of the arguments.


```python
x = mxp.asarray([1, 2, 3, 4], mask=[1, 0, 1, 0])
y = mxp.asarray([5, 6, 7, 8], mask=[1, 1, 0, 0])
x + y
```




    MArray(array([ 6,  8, 10, 12]), array([ True,  True,  True, False]))




```python
mxp.pow(y, x)
```




    MArray(array([   5,   36,  343, 4096]), array([ True,  True,  True, False]))



Note that `np.ma` automatically masks non-finite elements produced during calculations.


```python
import numpy

x = numpy.ma.masked_array(0, mask=False)
with numpy.errstate(divide='ignore', invalid='ignore'):
    y = [1, 0] / x
y
```




    masked_array(data=[--, --],
                 mask=[ True,  True],
           fill_value=1e+20,
                dtype=float64)



`MArray` *does not* follow this convention.


```python
x = mxp.asarray(0, mask=False)
with numpy.errstate(divide='ignore', invalid='ignore'):
    y = [1, 0] / x
y
```




    MArray(array([inf, nan]), array([False, False]))



This is because masked elements are often used to represent *missing* data, and the results of these operations are not missing. If this does not suit your needs, mask out data according to your requirements after performing the operation.


```python
x = mxp.asarray(0, mask=False)
with numpy.errstate(divide='ignore', invalid='ignore'):
    y = [1, 0] / x
mxp.asarray(y.data, mask=xp.isnan(y.data))
```




    MArray(array([inf, nan]), array([False,  True]))



#### Linear Algebra Functions
As usual, linear algebra functions and operators treat masked elements as though they don't exist.


```python
x = mxp.asarray([1, 2, 3, 4], mask=[1, 0, 1, 0])
y = mxp.asarray([5, 6, 7, 8], mask=[1, 1, 0, 0])
x @ y  # the last elements of the arrays, 4 and 8, are the only non-masked elements
```




    MArray(array(32), array(False))



The exception is `matrix_transpose`, which transposes the data and the mask.


```python
x = mxp.asarray([[1, 2], [3, 4]], mask=[[1, 1], [0, 0]])
x
```




    MArray(
        array([[1, 2],
           [3, 4]]),
        array([[ True,  True],
           [False, False]])
    )




```python
mxp.matrix_transpose(x)
```




    MArray(
        array([[1, 3],
           [2, 4]]),
        array([[ True, False],
           [ True, False]])
    )



#### Conclusion
While this tutorial is not exhaustive, we hope it is sufficient to allow you to predict the results of operations with `MArray`s and use them to suit your needs. If you'd like to see this tutorial extended in a particular way, please [open an issue](https://github.com/mdhaber/marray/issues)!
