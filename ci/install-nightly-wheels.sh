#!/usr/bin/env bash

python -m pip uninstall numpy

python -m pip install --upgrade --no-deps --pre \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    numpy
