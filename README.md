Installation
------------

If you want to use the GPU, you need a CUDA runtime, NVCC compiler and pycuda installed, but this is not required.

Then, just

    git clone https://gitlab.com/quantumsim/quantumsim

then

    pip install quantumsim/ [--user]

to install into the current environment or user site-packages,
or

    pip install -e quantumsim

to install in 'editable' mode, so that packages are imported from this directory.

To run the test suite, enter the directory and run py.test:

    cd quantumsim
    py.test

If you do not have pycuda available, GPU related tests will be skipped.


License
-------

This work is distributed under the GNU GPLv3. See LICENSE.txt.
(c) 2018 Quantumsim Authors
