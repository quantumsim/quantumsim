#!/usr/bin/env python

import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(name='quantumsim',
      version='0.1',
      description='Simulation of quantum circuits under somewhat realistic condititons',
      author='Brian Tarasinski et al',
      author_email='brianzi@physik.fu-berlin.de',
      ext_modules=cythonize(Extension("dmcpu", ["quantumsim/dmcpu.pyx"])),
      packages=['quantumsim'],
      ext_package='quantumsim',
      data_files=[('pycudakernels', ['src/primitives.cu'])]
      )
