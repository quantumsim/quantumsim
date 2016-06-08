import dm10

import numpy as np


n = 10

a = np.random.random(((2**n), (2**n)))*1j
a += np.random.random(((2**n), (2**n)))

a = np.dot(a, a.transpose().conj())

a = a/np.trace(a)

dm = dm10.Density(n, data = a)


bit = 0

for i in range(100):
    dm.hadamard(bit)
    dm.cphase(bit, bit+1)
    dm.rotate_y(bit, 0, 1)
    dm.amp_ph_damping(bit, 0.001, 0.001)
