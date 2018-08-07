#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='quantumsim',
    version='0.1',
    description=(
        'Simulation of quantum circuits under somewhat realistic condititons'
    ),
    author='Brian Tarasinski et al',
    author_email='brianzi@physik.fu-berlin.de',
    packages=find_packages('.'),
    ext_package='quantumsim',
    package_data={
        # all Cuda files we can find
        '': '*.cu',
    },
    install_requires=[
        "pytools",
        "numpy>=1.12",
        "pytest",
        "matplotlib",
        "parsimonious"
    ]
)
