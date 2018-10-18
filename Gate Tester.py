 # -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:58:37 2018

@author: dript
"""

import numpy as np
import quantumsim.circuit
from quantumsim.circuit import Circuit
from quantumsim.circuit import uniform_noisy_sampler

c1 = Circuit(title="c1")
c1.add_qubit("1")
c1.add_qubit("2")

c1.add_gate(quantumsim.circuit.RotateArb("1",time=0,nx=0,ny=1,nz=0,theta=np.pi))


sdm1 = quantumsim.sparsedm.SparseDM(c1.get_qubit_names())


c1.apply_to(sdm1)

print(sdm1.full_dm.to_array().round(3))
#print(np.dot(sdm1.full_dm.dm.ravel(),sdm4.full_dm.dm.ravel()))
