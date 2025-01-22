<a href="https://mdhaber.github.io/marray">
  <img src="https://raw.githubusercontent.com/mdhaber/marray/main/mybook/logo.svg" alt="Logo. The MArray logo is a nod to NumPy's logo, but MArray is not affiliated with the NumPy project." width="110" height="110" align="left" />
</a>

[![PyPI Downloads](https://img.shields.io/pypi/dm/marray.svg?label=Pypi%20downloads)](https://pypi.org/project/marray/)

MArray adds masks to your favorite
[Python Array API Standard compatible](https://data-apis.org/array-api/latest/)
array library.

Install with `pip`:

```shell
pip install marray
```

or `conda`:

```shell
conda install -c conda-forge marray-python
```

Use the `from...import...as` syntax to get a masked array namespace.

```python3
# use with any Array API compatible library, installed separately
from marray import numpy as mxp
import numpy as xp  # optional (if the non-masked namespace is desired)
```

The resulting `mxp` namespace has all the features of `xp` that are specified
in the Array API standard, but they are modified to be mask-aware. Typically, the
signatures of functions in the `mxp` namespace match those in the standard;
the one notable exception is the addition of a `mask` keyword argument of `asarray`.

```python3
mxp.asarray([1, 2, 3], mask=[False, True, False])
# MArray(array([1, 2, 3]), array([False,  True, False]))
```

In the spirit of the [Zen of Python](https://peps.python.org/pep-0020/), this is the one
and only obvious way to set the mask of an array.

Documentation provided by attributes of `xp` are exposed in the `mxp`
namespace and are accessible via `help`. For more information, please see
[the tutorial](https://mdhaber.github.io/marray/tutorial.html).
