import sparsedm
import numpy as np

sdm = sparsedm.SparseDM(["D1", "D2", "D3", "A1", "A2"])

for bit in sdm.classical:
    sdm.classical[bit] = 1

sdm.classical["D3"] = 0

print(sdm.classical)

for t in range(20):
    sdm.hadamard("A1") 
    sdm.cphase("A1", "D1")
    sdm.cphase("A1", "D2")
    sdm.hadamard("A1")

    sdm.amp_ph_damping("A1", 0.1, 0)

    # natural sampling
    r = np.random.random()
    p0, p1 = sdm.peak_measurement("A1")
    if p0/(p0 + p1) > r:
        sdm.project_measurement("A1", 0)
    else:
        sdm.project_measurement("A1", 1)

    sdm.hadamard("A2") 
    sdm.cphase("A2", "D2")
    sdm.cphase("A2", "D3")
    sdm.hadamard("A2")
    sdm.amp_ph_damping("A2", 0.1, 0)

    r = np.random.random()
    p0, p1 = sdm.peak_measurement("A2")
    if p0/(p0 + p1) > r:
        sdm.project_measurement("A2", 0)
    else:
        sdm.project_measurement("A2", 1)

    print(sdm.classical, sdm.trace())
