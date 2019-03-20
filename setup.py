#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='quantumsim',
    version='0.2.0',
    description=(
        'Simulation of quantum circuits under somewhat realistic condititons'
    ),
    author='Brian Tarasinski',
    author_email='brianzi@physik.fu-berlin.de',
    packages=find_packages('.'),
    ext_package='quantumsim',
    package_data={
        # all Cuda and json files we can find
        '': ['*.cu', '*.json'],
    },
    install_requires=[
        "pytools",
        "numpy>=1.12",
        "pytest",
        "matplotlib",
        "parsimonious",
    ],
    extras_require={
        'cuda': [
            'pycuda',
        ],
    },
)
