import numpy as np


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
    assert np.allclose(ptm[:, 0], [1, 0, 0, 0])
    t = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)]])

    return np.dot(t, np.dot(ptm, t))


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
