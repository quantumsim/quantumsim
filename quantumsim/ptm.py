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

_ptm_basis_vectors_cache = {}

def general_ptm_basis_vector(n):
    """
    The vector of 'Pauli matrices' in dimension n.
    First the n diagonal matrices, then 
    the off-diagonals in x-like, y-like pairs
    """

    if n in _ptm_basis_vectors_cache:
        return _ptm_basis_vectors_cache[n]
    else:

        basis_vector = []

        for i in range(n):
            v = np.zeros((n, n), np.complex)
            v[i, i] = 1
            basis_vector.append(v)

        for i in range(n):
            for j in range(i):
                #x-like
                v = np.zeros((n, n), np.complex)
                v[i, j] = np.sqrt(0.5) 
                v[j, i] = np.sqrt(0.5) 
                basis_vector.append(v)

                #y-like
                v = np.zeros((n, n), np.complex)
                v[i, j] = 1j*np.sqrt(0.5) 
                v[j, i] = -1j*np.sqrt(0.5) 
                basis_vector.append(v)

        basis_vector = np.array(basis_vector)

        _ptm_basis_vectors_cache[n] = basis_vector

    return basis_vector


def to_0xy1_basis(ptm, general_basis=False):
    """Transform a Pauli transfer in the "usual" basis (0xyz) [1],
    to the 0xy1 basis which is required by sparsesdm.apply_ptm.

    If general_basis is True, transform to the 01xy basis, which is the 
    two-qubit version of the general basis defined by ptm.general_ptm_basis_vector().

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
    result = np.dot(
        basis_transformation_matrix, np.dot(
            ptm, basis_transformation_matrix))

    if general_basis:
        result = result[[0, 3, 1, 2], :][:, [0, 3, 1, 2]]

    return result



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

def hadamard_ptm(general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary Hadamard (Rotation around the (x+z)/sqrt(2) axis by π).
    """
    u = np.array([[1, 1], [1, -1]])*np.sqrt(0.5)
    return single_kraus_to_ptm(u, general_basis)


def amp_ph_damping_ptm(gamma, lamda, general_basis=False):
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
    return to_0xy1_basis(ptm, general_basis)

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


def rotate_x_ptm(angle, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the x-axis by angle.
    """
    ptm = np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])

    return to_0xy1_basis(ptm, general_basis)


def rotate_y_ptm(angle, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the y-axis by angle.
    """
    ptm = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])

    return to_0xy1_basis(ptm, general_basis)


def rotate_z_ptm(angle, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the z-axis by angle.
    """
    ptm = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    return to_0xy1_basis(ptm, general_basis)


def single_kraus_to_ptm_general(kraus):
    d = kraus.shape[0]
    assert kraus.shape == (d, d)

    st = general_ptm_basis_vector(d)

    return np.einsum("xab, bc, ycd, ad -> xy", st, kraus, st, kraus.conj()).real

def single_kraus_to_ptm(kraus, general_basis=False):
    """Given a Kraus operator in z-basis, obtain the corresponding single-qubit ptm in 0xy1 basis"""
    if general_basis:
        st = general_ptm_basis_vector(2)
    else:
        st = single_tensor
    return np.einsum("xab, bc, ycd, ad -> xy", st, kraus, st, kraus.conj()).real

def double_kraus_to_ptm(kraus, general_basis=False):
    if general_basis:
        st = general_ptm_basis_vector(2)
    else:
        st = single_tensor

    dt = np.kron(st, st)

    return np.einsum("xab, bc, ycd, ad -> xy", dt, kraus, dt, kraus.conj()).real
