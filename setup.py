#!/usr/bin/env python

import setuptools
from distutils.core import setup


setup(name='quantumsim',
      version='0.1',
      description='Simulation of quantum circuits under somewhat realistic condititons',
      author='Brian Tarasinski et al',
      author_email='brianzi@physik.fu-berlin.de',
      packages=['quantumsim'],
      ext_package='quantumsim',
      data_files=[('pycudakernels', ['quantumsim/primitives.cu'])],
      requires=["pytools", "numpy(>=1.12)", "pytest", "matplotlib"]
      )
