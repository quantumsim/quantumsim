#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='qs2',
    version='1.0.0-dev0',
    description=(
        'Simulation of quantum circuits under somewhat realistic condititons'
    ),
    author='Quantumsim Authors',
    author_email='brianzi@physik.fu-berlin.de',
    packages=find_packages('.'),
    ext_package='qs2',
    package_data={
        # all Cuda and json files we can find
        '': ['*.cu', '*.json'],
    },
    install_requires=list(open('requirements.txt').read().strip().split('\n')),
)
