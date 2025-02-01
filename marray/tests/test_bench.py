import numpy as xp
from marray import numpy as mxp
import time, json
from test_marray import (arithmetic_unary, elementwise_unary, elementwise_binary,
                         statistical_array, utility_array, searching_array,
                         comparison_binary, get_arrays)

seed = 64379182864537915

data = {}

class Timeit:
    def __init__(self, f_name, array_type):
        self.f_name = f_name
        self.array_type = array_type
        data[f_name] = data.get(f_name, {})

    def __enter__(self):
        self.tic = time.perf_counter_ns()

    def __exit__(self, type_, value, traceback):
        self.toc = time.perf_counter_ns()
        data[self.f_name][self.array_type] = self.toc - self.tic

for n, fdict in [(1, arithmetic_unary), (2, comparison_binary)]:
    for f_name, f in fdict.items():
        marrays, masked_arrays, seed = get_arrays(n, shape=(1000, 1000), ndim=2,
                                                  dtype='float64', xp=xp, seed=seed)

        with Timeit(f_name, 'MArray      '):
            res = f(*marrays)
        with Timeit(f_name, 'masked_array'):
            ref = f(*masked_arrays)

for n, flist in [(1, elementwise_unary), (2, elementwise_binary)]:
    for f_name in flist:
        marrays, masked_arrays, seed = get_arrays(n, shape=(1000, 1000), ndim=2,
                                                  dtype='float64', xp=xp, seed=seed)

        f = getattr(mxp, f_name)
        with Timeit(f_name, 'MArray      '):
            res = f(*marrays)

        f = getattr(xp, f_name)
        with Timeit(f_name, 'masked_array'):
            ref = f(*masked_arrays)

for f_name in (statistical_array + utility_array + searching_array
               + ["sort", "argsort"]):
    marrays, masked_arrays, seed = get_arrays(1, shape=(1000, 1000), ndim=2,
                                              dtype='float64', xp=xp, seed=seed)

    f = getattr(mxp, f_name)
    with Timeit(f_name, 'MArray      '):
        res = f(*marrays, axis=-1)

    f = getattr(xp, f_name)
    with Timeit(f_name, 'masked_array'):
        ref = f(*masked_arrays, axis=-1)

# print(json.dumps(data, indent=4))
for fun_name, d in data.items():
    print(f"{fun_name}: {d["MArray      "]/d["masked_array"]}")
