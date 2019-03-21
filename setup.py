#!/usr/bin/env python

from setuptools import setup, find_packages


# Loads version.py module without importing the whole package.
def get_version_and_cmdclass(package_path):
    import os
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('version',
                                   os.path.join(package_path, '_version.py'))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass('qs2')

setup(
    name='qs2',
    url='https://quantumsim.gitlab.io/',
    version=version,
    cmdclass=cmdclass,
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
    extras_require={
        'cuda': list(open('requirements-gpu.txt').read().strip().split('\n'))
    }
)
