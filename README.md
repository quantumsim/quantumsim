The plan:

We hold the full density matrix of the data qbits in memory, which is about two megabyte (so not much).


We add one ancilla to measure a syndrome, then trace it out again to measure it; then add the next ancilla etc.


We do NOT precompile any operations, as i believe that block application is easier than actually 
multiplying with matrices that have been sparsified by hand. The structure is too strong in this one.


We thus need the following primitives, which will be implemented by hand as kernels:

At the beginning and end of each syndrome measurement:
  - Add ancilla qbit, taking a 2^9 x 2^9 density matrix (dm9) into a 2^10 x 2^10 density matrix (dm10)
  - trace the ancilla out again, obtaining two dm9's (we can hold both of them or throw one away, to explore the statistical tree)

For a single measurement (all acting on one dm10):
  - Idle gate on a single qbit (asymmetric t1, t2 decay)
  - cphase gate on any qbit and the ancilla (maybe noisy at some point)
  - hadamard on any qbit (maybe noisy at some point)
  - (maybe pi pulses etc)

each of these I expect to take about 500us on a tesla (thats what dgemm(dm10, dm10) takes), 
and we need about 1000 of them per round; thus 500ms per round?



