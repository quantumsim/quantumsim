# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

"""functions for generating primitive PTM instances"""

import numpy as np

from .ptm import(
    ConjunctionPTM,
    TwoKrausPTM,
    ProductPTM)


class CPhaseRotationPTM(TwoKrausPTM):
    def __init__(self, angle=np.pi):
        u = np.diag([1, 1, 1, np.exp(1j * angle)]).reshape(2, 2, 2, 2)
        super().__init__(u)


class RotateXPTM(ConjunctionPTM):
    def __init__(self, angle):
        s, c = np.sin(angle / 2), np.cos(angle / 2)
        super().__init__([[c, -1j * s], [-1j * s, c]])


class RotateYPTM(ConjunctionPTM):
    def __init__(self, angle):
        s, c = np.sin(angle / 2), np.cos(angle / 2)
        super().__init__([[c, -s], [s, c]])


class RotateZPTM(ConjunctionPTM):
    def __init__(self, angle):
        z = np.exp(-.5j * angle)
        super().__init__([[z, 0], [0, z.conj()]])


class AmplitudePhaseDampingPTM(ProductPTM):
    def __init__(self, gamma, lamda):
        e0 = [[1, 0], [0, np.sqrt(1 - gamma)]]
        e1 = [[0, np.sqrt(gamma)], [0, 0]]
        amp_damp = ConjunctionPTM(e0) + ConjunctionPTM(e1)

        e0 = [[1, 0], [0, np.sqrt(1 - lamda)]]
        e1 = [[0, 0], [0, np.sqrt(lamda)]]
        ph_damp = ConjunctionPTM(e0) + ConjunctionPTM(e1)

        super().__init__([amp_damp, ph_damp])