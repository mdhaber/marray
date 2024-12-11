# marray

Masked versions of Array API compatible arrays.

The only public function is `get_namespace`:

```python3
import numpy as xp  # use any Array API compatible library
import marray
mxp = marray.get_namespace(xp)
```

The resulting `mxp` namespace has all features of the `xp` are specified
in the Array API standard, modified to be mask-aware. Typically, the
signatures of functions in the `mxp` namespace match those in the `xp` namespace;
the one notable exception is the addition of a `mask` keyword argument of `asarray`.

```python3
mxp.asarray([1, 2, 3], mask=[False, True, False])
# MArray(array([1, 2, 3]), array([False,  True, False]))
```

In the spirit of the [Zen of Python](https://peps.python.org/pep-0020/), this is the one
and only obvious way to set the mask of an array.

Documentation provided by attributes of `xp` attributes are exposed in the `mxp`
namespace and are accessible via `help`. For more information, please see 
[the tutorial](https://colab.research.google.com/drive/1LaZCK3jvnf40qEjWEhqhhc5e8IWy7vuo?usp=sharing)!
