#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize

setup(name='quantumsim',
    version='0.1',
    description='Simulation of quantum circuits under somewhat realistic condititons',
    author='Brian Tarasinski et al',
    author_email='brianzi@physik.fu-berlin.de',
    ext_modules = cythonize("dmcpu.pyx"),
    py_modules=['dmcpu', 'dm10', 'sparsedm', 'circuit']

    data_files=[('kernelsource', ['primitives.cu'])]
    )
