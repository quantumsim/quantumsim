TODOs
-----

  - Implement primitives in cython as well, let the dm10 module select which of the two implementations to use.
    - Maybe even a pure numpy implementation (using shitloads of np.einsum)
    - Make the Density class use cython for small dms and switch to cuda when large, automatically
  - A Circuit class which allows writing down gates as qasm-like strings (but with time informations):
    - Integrated test on 3qb repetition code
      - clean
      - with decay


The plan
========

We hold the full density matrix of the data qubits in memory, which is about two megabyte (so not much).

The density matrix is hermitian, so we only need to store one half.

However, in order to do address magic, we will store it in full form (memory is not an issue)
but set the lower triangle to "don't care" and only update the upper half.

This means that we typically run grids that are twice too big. Often we can use the
second half to do the imaginary parts instead.
We also store as in numpy format, that is, real and imaginary part next to each other.
Maybe we want to change this at some point.
The matrix element Re(dm[x,y]) is actually stored at position

    dm[x,y] = dm[(x<<noqbits|y)<<1].

We add one ancilla to measure a syndrome, then trace it out again to measure it; then add the next ancilla etc.

We do NOT precompile any operations, as i believe 
that block application is easier than actually 
multiplying with matrices that have been 
sparsified by hand. The structure is too strong in this one.

Primitives
==========

We thus need the following primitives, which will be implemented by hand as kernels:

At the beginning and end of each syndrome measurement:
  - Add ancilla qbit, taking a 2^9 x 2^9 density matrix (dm9) into a 2^10 x 2^10 density matrix (dm10)
  - trace the ancilla out again, obtaining two dm9's (we can hold both of them or throw one away, to explore the statistical tree)

For a single measurement (all acting on one dm10):
  - Idle gate on a single qbit (amplitude and phase damping, i.e. t1, t2 decay)
  - cphase gate on any qbit and the ancilla (maybe noisy at some point)
  - hadamard on any qbit (maybe noisy at some point)
  - (maybe pi pulses etc)

each of these I expect to take about 500us on a tesla (thats what dgemm(dm10, dm10) takes), 
and we need about ~500 of them per round; thus 250ms per round?



Wrappers
========

The data structure exposing the above primitives is called `dm10.Density`.

`d = dm10.Density(10)` creates a density matrix with 10 qubits, 
allocating memory on the device.  
One can operate on it like

        d.cphase(bit0=2, bit1=3) 
        d.amp_phase_damping(bit=0, gamma=0.1, lamda=1)
        d.hadamard(bit=4)`
        t = d.trace()

The bits are labelled 0 to 9.

Adding an addition of a new ancilla qubit is done by

        d_new = d.add_ancilla(bit=10, anc_state=0)

This creates a new density matrix of larger size, including the allocation of new memory.
(There is some optimization possible here, maybe add some sort of in-place inflation).

The measurement of an ancilla is done by

      p0, dm0, p1, dm1 = d.measure_ancilla(bit=9)

This creates two new density matrices (including newly allocated memory)
and returns them, together with their traces.
Again, this could be done with reuse of allocated memory, so there is some optimization 
possible here.

It is important to note that dm0 and dm1 are not normalized to trace one, they are the subblocks of the original density matrix.

Sparse density matrix
=====================

In order to keep track of which ancillas are in the density matrix at any moment,
there is a second level of abstraction, implemented in `sparsedm.SparseDM`.

This object keeps track of a set of qubits, which can either be 
"quantum", that is, part of a dm10.Density; or "classical", that is, be
in a pure state as a condition for the dm10.Density.

We create one by
      
      sdm = sparsedm.SparseDM(["Q1", "Q2", "Q3"])


These are the names of all qubits. Anything hashable (strings, numbers etc) 
can be used as names for quibits.

In the beginning, all qubits are classical and in the ground state. We can see and change the 
state of the classical qubits by inspecting the dict `sdm.classical`.

We can then apply gates to qubits, e.g.

      sdm.cphase("Q1", "Q3")
      sdm.hadamard("Q2")

When this happens, the qubit is added to the dm10.Density and removed from sdm.classical. 
Because a dm10.Density always enumerates qubits as zero based integers, there is a dict
`sdm.idx_in_full_dm`, mapping the qubit names to indices.

Measurement is done in two steps. First, we can look at the probabilities for 
the outcome of a measurement:

      p0, p1 = sdm.peak_measurement("Q1")

Note that these two probabilities do not have to sum up to one, they are the total 
probability of this outcome, taking into account the selected outcomes of all previous measurements.

Then we can select the outcome of the measurement arbitrarily:

      sdm.project_measurement("Q1", state=0)

This will project dm10.Density to the subblock 
with this outcome and add it back to sdm.classical with state 0.
Again, note that the trace of the dm10.Density is not renormalized; this means that the 
trace gives the total probability with which the selected outcome of the measeruments 
actually takes place. The trace can be obtained using `sdm.trace()`.

```python

```
