# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

"""functions for generating primitive PTM instances"""

import numpy as np

from qs2.operators import(
    ConjunctionPTM,
    TwoKrausPTM,
    ProductPTM)

def amplitude_phase_damping_ptm(gamma, lamda):
    e0 = [[1, 0], [0, np.sqrt(1 - gamma)]]
    e1 = [[0, np.sqrt(gamma)], [0, 0]]
    amp_damp = ConjunctionPTM(e0) + ConjunctionPTM(e1)

    e0 = [[1, 0], [0, np.sqrt(1 - lamda)]]
    e1 = [[0, 0], [0, np.sqrt(lamda)]]
    ph_damp = ConjunctionPTM(e0) + ConjunctionPTM(e1)

    return ProductPTM([amp_damp, ph_damp])

def cphase_rotation_ptm(angle=np.pi):
    u = np.diag([1, 1, 1, np.exp(1j * angle)]).reshape(2, 2, 2, 2)
    return TwoKrausPTM(u)

def rotate_x_ptm(angle):
    s, c = np.sin(angle / 2), np.cos(angle / 2)
    return ConjunctionPTM([[c, -1j * s], [-1j * s, c]])

def rotate_y_ptm(angle):
    s, c = np.sin(angle / 2), np.cos(angle / 2)
    return ConjunctionPTM([[c, -s], [s, c]])

def rotate_z_ptm(angle):
    z = np.exp(-.5j * angle)
    return ConjunctionPTM([[z, 0], [0, z.conj()]])
