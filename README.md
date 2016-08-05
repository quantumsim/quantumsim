Installation
------------

You need cython installed. If you want to use the GPU, you need a CUDA runtime, NVCC compiler and pycuda installed, but this is not required.

Just

    git clone https://github.com/brianzi/quantumsim

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


Overview and usage
==================

To obtain an overview over the capabilities of the package from a user perspective,
have a look at `Introduction.ipynb`.


The architecture 
========

The backend of this package is a full density matrix of n qubits, stored on a GPU.
(a CPU backend is also implemented, but too slow to be useful for big n at the moment).

Data format
-----------

The density matrix is not stored as a hermitian complex matrix, a different basis is used.
In the single qubit case, we can define the four basis matrices:

  s0 = [1 0]  sx = [0 1]
       [0 0]       [1 0] / sqrt(2)

  s1 = [0 0]  sy = [0 -i]
       [0 1]       [i  0] / sqrt(2)

We can then store the density matrix `R` as the four real numbers `r(a) = Tr(s_a R)`,
where `a` takes the four values (0, x, y, 1). We thus refer to this basis as the `0xy1` basis.
It is not as often used in the literature as the similar `1xyz` or Pauli basis, but has the advantage 
that the diagonal immediately represents probabilities of local measurement outcomes.

For several qubits, we use the natural extension, `r(abc...) = Tr((s_a x s_b x
s_c x ...) R)`, where `x` denotes the Kronecker product and `s_a, s_b, s_c, ...` 
act on the 0th, 1st, 2nd, ... qubit.  By letting 
    0 -> 00, x -> 01, y -> 10, 1 -> 11
in the string `abc...`, we immediately obtain an address in binary
for storing of the numbers r(abc...)

Primitives
----------

The backend supports the following primitives, which are implemented as custom CUDA kernels ("GPU programs"):


- Apply any single-qubit channel to any of the qubits. The channels are represented as Pauli transfer matrices.
- Add a new qubit in the ground state to the density matrix, increasing its size by a factor 4.
- Project a qubit to either 0 or 1 and remove it from the density matrix. This reduces the 
size of the density matrix by a factor 4 and reduces its trace. 
The new trace is the probabilty for this projection to happen during the experiment.
- Apply a perfect C-Phase gate between any two qubits.


More primitives, especially general two-qubit gates, will be added when needed.

Wrappers
--------

The data structure exposing the above primitives to python is called `dm10.Density`.

Sparse density matrix
---------------------

In order to keep track of which ancillas are in the density matrix at any moment,
there is a second level of abstraction, implemented in `sparsedm.SparseDM`.

This object represents the states of a set of named qubits, which can either be 
"quantum", that is, part of a dm10.Density; or "classical", that is, not entangled
with any other qubit.
Anything hashable (strings, numbers etc) can be used as names for quibits.


In a newly generated SparseDM, all qubits are classical and in the ground state.
Its state can be changed by applying gates, e.g.

      sdm.cphase("Q1", "Q3")
      sdm.hadamard("Q2")

When this happens, the qubit is added to the internal dm10.Density and removed from the classical qubits.
Because a dm10.Density always enumerates qubits as zero based integers, there is a dict
`sdm.idx_in_full_dm`, mapping the qubit names to indices.

Measurement sampling can be done in two steps. First, we can obtain the probabilities for 
the outcome of a measurement:

      p0, p1 = sdm.peak_measurement("Q1")

Note that these two probabilities do not have to sum up to one, they are the total 
probability of this outcome, taking into account the selected outcomes of all previous measurements.

Then we can select the outcome of the measurement arbitrarily, for instance selected using p0, p1 and a source of randomness:

      sdm.project_measurement("Q1", state=0)

This will project dm10.Density to the subblock 
with this outcome and add it back to sdm.classical with state 0.
Again, note that the trace of the dm10.Density is not renormalized; this means that the 
trace gives the total probability with which the selected outcome of the measeruments 
actually takes place. The trace can be obtained using `sdm.trace()`.


Circuits
========

Finally there is a python module representing experimental protocols ("circuits") 
in terms of gates and measurements, their timing information, and automatically added decay channels. 

After defining such a circuit, it can be applied to a SparseDM in one line of code.


License
-------

This work is distributed under the GNU GPLv3. See LICENSE.txt.
(c) 2016 Brian Tarasinski
