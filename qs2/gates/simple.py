# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

"""functions for generating simple gates - i.e. those
that can be described in a few lines."""

from qs2.operators import RotationGate
from qs2.gates import (
    amplitude_phase_damping_ptm,
    cphase_rotation_ptm,
    rotate_x_ptm,
    rotate_y_ptm,
    rotate_z_ptm)


def rotate_x_unitary(qubits, setup, angle=None):
    """
    Error-free rotation around the X axis

    sargs:
    ------
    None

    Params
    ------
    qubits : str
        The qubits that this gate acts upon
    setup : quantumsim.setup.Setup object
        Setup file for this experiment
    """
    return RotationGate(
        qubits=qubits, angle=angle,
        setup=setup, ptm_function=rotate_x_ptm)

def rotate_y_unitary(qubits, setup, angle=None):
    """
    Error-free rotation around the Y axis

    sargs:
    ------
    None

    Params
    ------
    qubits : str
        The qubits that this gate acts upon
    setup : quantumsim.setup.Setup object
        Setup file for this experiment
    """
    return RotationGate(
        qubits=qubits, angle=angle,
        setup=setup, ptm_function=rotate_y_ptm)

def rotate_z_unitary(qubits, setup, angle=None):
    """
    Error-free rotation around the Z axis

    sargs:
    ------
    None

    Params
    ------
    qubits : str
        The qubits that this gate acts upon
    setup : quantumsim.setup.Setup object
        Setup file for this experiment
    """
    return RotationGate(
        qubits=qubits, angle=angle,
        setup=setup, ptm_function=rotate_z_ptm)
