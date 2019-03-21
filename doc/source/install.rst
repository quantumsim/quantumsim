Installation
============

From source
-----------

If you want to install Quantumsim:

.. code-block:: bash

    git clone https://gitlab.com/quantumsim/quantumsim.git
    cd quantumsim
    pip install .  # if you don't need GPU support
    # or
    pip install numpy  # if it is not installed already
    pip install .[cuda]  # if you need GPU support

Note the need to install numpy in advance: `pycuda` installer fails, if it is
not installed already.