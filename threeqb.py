import sparsedm
import numpy as np

import circuit as c


qubit_names = ["D1", "D2", "D3", "A1", "A2"]
sdm = sparsedm.SparseDM(qubit_names)

for bit in sdm.classical:
    sdm.classical[bit] = 1

sdm.classical["D3"] = 0

print(sdm.classical)



c = circuit.Circuit()

for qb in qubit_names:
    c.add_qubit(qb, 30000, 50000)

c.add_gate(circuit.Hadamard("A1", time=0))
c.add_gate(circuit.Hadamard("A2", time=0))

c.add_gate()




