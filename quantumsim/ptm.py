# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np

"The transformation matrix between the two basis. Its essentially a Hadamard, so its its own inverse."
basis_transformation_matrix = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)]])

single_tensor = np.array([[[1, 0], [0, 0]],
                          np.sqrt(0.5) * np.array([[0, 1], [1, 0]]),
                          np.sqrt(0.5) * np.array([[0, -1j], [1j, 0]]),
                          [[0, 0], [0, 1]]])

double_tensor = np.kron(single_tensor, single_tensor)


def to_0xy1_basis(ptm):
    """Transform a Pauli transfer in the "usual" basis (0xyz) [1],
    to the 0xy1 basis which is required by sparsesdm.apply_ptm.

    ptm: The input transfer matrix in 0xyz basis. Can be 4x4, 4x3 or 3x3 matrix of real numbers.

         If 4x4, the first row must be (1,0,0,0). If 4x3, this row is considered to be omitted.
         If 3x3, the transformation is assumed to be unitary, thus it is assumed that
         the first column is also (1,0,0,0) and was omitted.

    [1] Daniel Greenbaum, Introduction to Quantum Gate Set Tomography, http://arxiv.org/abs/1509.02921v1
    """

    ptm = np.array(ptm)

    if ptm.shape == (3, 3):
        ptm = np.hstack(([[0], [0], [0]], ptm))

    if ptm.shape == (3, 4):
        ptm = np.vstack(([1, 0, 0, 0], ptm))

    assert ptm.shape == (4, 4)
    assert np.allclose(ptm[0, :], [1, 0, 0, 0])
    return np.dot(
        basis_transformation_matrix, np.dot(
            ptm, basis_transformation_matrix))


def to_0xyz_basis(ptm):
    """Transform a Pauli transfer in the 0xy1 basis [1],
    to the the usual 0xyz. The inverse of to_0xy1_basis.

    ptm: The input transfer matrix in 0xy1 basis. Must be 4x4.

    [1] Daniel Greenbaum, Introduction to Quantum Gate Set Tomography, http://arxiv.org/abs/1509.02921v1
    """

    ptm = np.array(ptm)
    if ptm.shape == (4, 4):
        trans_mat = basis_transformation_matrix
        return np.dot(trans_mat, np.dot(ptm, trans_mat))
    elif ptm.shape == (16, 16):
        trans_mat = np.kron(basis_transformation_matrix, basis_transformation_matrix)
        return np.dot(trans_mat, np.dot(ptm, trans_mat))
    else:
        raise ValueError("Dimensions wrong, must be one- or two Pauli transfer matrix ")




def hadamard_ptm():
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary Hadamard (Rotation around the (x+z)/sqrt(2) axis by π).
    """
    return np.array([[0.5, np.sqrt(0.5), 0, 0.5],
                     [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)],
                     [0, 0, -1, 0],
                     [0.5, -np.sqrt(0.5), 0, 0.5]], np.float64)


def amp_ph_damping_ptm(gamma, lamda):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing amplitude and phase damping with parameters gamma and lambda.
    (See Nielsen & Chuang for definition.)
    """
    ptm = np.array([
        [1, 0, 0, 0],
        [0, np.sqrt((1 - gamma) * (1 - lamda)), 0, 0],
        [0, 0, np.sqrt((1 - gamma) * (1 - lamda)), 0],
        [gamma, 0, 0, 1 - gamma]]
    )
    return to_0xy1_basis(ptm)

def gen_amp_damping_ptm(gamma_down, gamma_up):
    """Return a 4x4 Pauli transfer matrix  representing amplitude damping including an excitation rate gamma_up.
    """

    gamma = gamma_up + gamma_down
    p = gamma_down/(gamma_down+gamma_up)


    ptm = np.array([
        [1, 0, 0, 0],
        [0, np.sqrt((1 - gamma)), 0, 0],
        [0, 0, np.sqrt((1 - gamma)), 0],
        [(2*p - 1)*gamma, 0, 0, 1 - gamma]]
    )

    return to_0xy1_basis(ptm)

def dephasing_ptm(px, py, pz):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing dephasing (shrinking of the Bloch sphere along the principal axes),
    with different rates across the different axes.
    p_i/2 is the flip probability, so p_i = 0 corresponds to no shrinking, while p_i = 1 is total dephasing.
    """

    ptm = np.diag([1 - px, 1 - py, 1 - pz])
    return to_0xy1_basis(ptm)


def bitflip_ptm(p):
    ptm = np.diag([1 - p, 1, 1])
    return to_0xy1_basis(ptm)


def rotate_x_ptm(angle):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the x-axis by angle.
    """
    ptm = np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])

    return to_0xy1_basis(ptm)


def rotate_y_ptm(angle):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the y-axis by angle.
    """
    ptm = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])

    return to_0xy1_basis(ptm)


def rotate_z_ptm(angle):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the z-axis by angle.
    """
    ptm = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    return to_0xy1_basis(ptm)


def single_kraus_to_ptm(kraus):
    """Given a Kraus operator in z-basis, obtain the corresponding single-qubit ptm in 0xy1 basis"""
    return np.einsum("xab, bc, ycd, ad -> xy", single_tensor, kraus, single_tensor, kraus.conj()).real

def double_kraus_to_ptm(kraus):
    return np.einsum("xab, bc, ycd, ad -> xy", double_tensor, kraus, double_tensor, kraus.conj()).real
