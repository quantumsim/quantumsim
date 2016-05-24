import sparsedm


sdm = sparsedm.SparseDM(5)

for bit in sdm.classical:
    sdm.classical[bit] = 1

sdm.classical[2] = 0

print(sdm.classical)

for t in range(10):
    sdm.hadamard(4) 
    sdm.cphase(4, 2)
    sdm.cphase(4, 1)
    sdm.hadamard(4)

    # max likelihood sampling
    p0, p1 = sdm.peak_measurement(4)
    if p0 > p1:
        sdm.project_measurement(4, 0)
    else:
        sdm.project_measurement(4, 1)

    sdm.hadamard(3) 
    sdm.cphase(3, 1)
    sdm.cphase(3, 0)
    sdm.hadamard(3)

    p0, p1 = sdm.peak_measurement(3)
    if p0 > p1:
        sdm.project_measurement(3, 0)
    else:
        sdm.project_measurement(3, 1)

    print(sdm.classical)
