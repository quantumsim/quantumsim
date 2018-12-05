Installation
============

From source
-----------

If you want to install this branch of Quantumsim:

.. code-block:: bash

    git clone https://gitlab.com/quantumsim/quantumsim.git
    cd quantumsim
    git checkout qs2/master
    pip install numpy  # if it is not installed already
    pip install .  # if you don't need GPU support
    # or
    pip install .[gpu]  # if you need GPU support

Note the need to install numpy in advance: `pycuda` installer fails, if it is
not installed already.