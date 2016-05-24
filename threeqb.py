import sparsedm
import numpy as np

sdm = sparsedm.SparseDM(5)

for bit in sdm.classical:
    sdm.classical[bit] = 1

sdm.classical[2] = 0

print(sdm.classical)

for t in range(20):
    sdm.hadamard(4) 
    sdm.cphase(4, 2)
    sdm.cphase(4, 1)
    sdm.hadamard(4)

    sdm.amp_ph_damping(4, 0.1, 0)

    # natural sampling
    r = np.random.random()
    p0, p1 = sdm.peak_measurement(4)
    if p0/(p0 + p1) > r:
        sdm.project_measurement(4, 0)
    else:
        sdm.project_measurement(4, 1)

    sdm.hadamard(3) 
    sdm.cphase(3, 1)
    sdm.cphase(3, 0)
    sdm.hadamard(3)
    sdm.amp_ph_damping(3, 0.1, 0)

    r = np.random.random()
    p0, p1 = sdm.peak_measurement(3)
    if p0/(p0 + p1) > r:
        sdm.project_measurement(3, 0)
    else:
        sdm.project_measurement(3, 1)

    print(sdm.classical, sdm.trace())
