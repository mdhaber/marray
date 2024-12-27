# marray

`marray` adds masks to your favorite
[Python Array API Standard compatible](https://data-apis.org/array-api/latest/)
array library.

Install with `pip`.

```shell
pip install marray
```

The only public function is `get_namespace`:

```python3
import numpy as xp  # use any Array API compatible library, installed separately
import marray
mxp = marray.get_namespace(xp)
```

The resulting `mxp` namespace has all the features of `xp` that are specified
in the Array API standard, but they are modified to be mask-aware. Typically, the
signatures of functions in the `mxp` namespace match those in the `xp` namespace;
the one notable exception is the addition of a `mask` keyword argument of `asarray`.

```python3
mxp.asarray([1, 2, 3], mask=[False, True, False])
# MArray(array([1, 2, 3]), array([False,  True, False]))
```

In the spirit of the [Zen of Python](https://peps.python.org/pep-0020/), this is the one
and only obvious way to set the mask of an array.

Documentation provided by attributes of `xp` are exposed in the `mxp`
namespace and are accessible via `help`. For more information, please see
[the tutorial](https://colab.research.google.com/drive/1LaZCK3jvnf40qEjWEhqhhc5e8IWy7vuo?usp=sharing).
