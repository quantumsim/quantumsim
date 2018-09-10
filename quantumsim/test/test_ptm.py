import quantumsim.ptm as ptm
import numpy as np
from scipy.stats import unitary_group


# some states in 0xy1 basis
ground_state = np.array([1, 0, 0, 0])
excited_state = np.array([0, 0, 0, 1])

rng = np.random.RandomState(42)


def random_state():
    a = rng.rand(4)

    a /= a[0] + a[3]

    return a


class TestPTMBasisConversion:
    def test_convert_unity(self):
        assert np.allclose(ptm.to_0xy1_basis(np.eye(4)), np.eye(4))
        assert np.allclose(ptm.to_0xy1_basis(np.eye(3)), np.eye(4))

    def test_explicit_on_hadamard(self):
        s2 = np.sqrt(0.5)
        ptm_hadamard_0xy1_should_be = np.array(
            [[0.5, s2, 0, 0.5],
             [s2, 0, 0, -s2],
             [0, 0, -1, 0],
             [0.5, -s2, 0, 0.5]]
        )

        ptm3x3hadamard = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        ptm_hadamard = ptm.to_0xy1_basis(ptm3x3hadamard)
        assert np.allclose(ptm_hadamard_0xy1_should_be, ptm_hadamard)

        ptm4x3hadamard = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])
        ptm_hadamard = ptm.to_0xy1_basis(ptm4x3hadamard)
        assert np.allclose(ptm_hadamard_0xy1_should_be, ptm_hadamard)

        ptm4x4hadamard = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])
        ptm_hadamard = ptm.to_0xy1_basis(ptm4x4hadamard)
        assert np.allclose(ptm_hadamard_0xy1_should_be, ptm_hadamard)

    def test_inversion(self):
        p = np.random.random((4, 4))

        p[0, :] = np.array([1, 0, 0, 0])

        print(p)

        assert np.allclose(p, ptm.to_0xyz_basis(ptm.to_0xy1_basis(p)))

    def test_random_singlequbit(self):
        for i in range(10):
            kraus = unitary_group.rvs(2, random_state=rng)
            p = ptm.single_kraus_to_ptm(kraus)
            assert np.allclose(ptm.to_0xy1_basis(ptm.to_0xyz_basis(p)), p)

    def test_random_twoqubit(self):
        for i in range(10):
            kraus = unitary_group.rvs(4, random_state=rng)
            p = ptm.double_kraus_to_ptm(kraus)
            assert np.allclose(ptm.to_0xy1_basis(ptm.to_0xyz_basis(p)), p)


class TestRotations:
    def test_xyz(self):
        ptm_x = ptm.rotate_x_ptm(np.pi/2)
        ptm_z = ptm.rotate_x_ptm(np.pi/2)
        ptm_y = ptm.rotate_x_ptm(-np.pi/2)

        ptm_xyz = np.dot(ptm_y, np.dot(ptm_z, ptm_x))

        assert np.allclose(ptm_xyz, ptm_z)

    def test_power(self):
        ptm_x = ptm.rotate_x_ptm(2*np.pi/7)
        ptm_x7 = np.linalg.matrix_power(ptm_x, 7)
        assert np.allclose(ptm_x7, np.eye(4))

        ptm_y = ptm.rotate_y_ptm(2*np.pi/7)
        ptm_y7 = np.linalg.matrix_power(ptm_y, 7)
        assert np.allclose(ptm_y7, np.eye(4))

        ptm_z = ptm.rotate_z_ptm(2*np.pi/7)
        ptm_z7 = np.linalg.matrix_power(ptm_z, 7)
        assert np.allclose(ptm_z7, np.eye(4))

    def test_excite_deexcite_ground_state(self):

        ptm_x = ptm.rotate_x_ptm(np.pi)
        ptm_y = ptm.rotate_y_ptm(np.pi)
        ptm_z = ptm.rotate_z_ptm(np.pi)

        state = ground_state
        state = np.dot(ptm_x, state)
        state = np.dot(ptm_y, state)
        state = np.dot(ptm_z, state)
        state = np.dot(ptm_x, state)
        state = np.dot(ptm_y, state)
        state = np.dot(ptm_z, state)

        assert np.allclose(state, ground_state)

    def test_xyx_ground_state(self):

        state = ground_state

        state = ptm.rotate_x_ptm(np.pi/2).dot(state)
        state = ptm.rotate_y_ptm(np.pi/2).dot(state)
        state = ptm.rotate_x_ptm(-np.pi/2).dot(state)

        assert np.allclose(state, ground_state)


class TestAmpPhaseDamping:
    def test_does_nothing_to_ground_state(self):
        p = ptm.amp_ph_damping_ptm(0.23, 0.42)

        assert np.allclose(p.dot(ground_state), ground_state)

    def test_strong_damping_gives_ground_state(self):
        r = random_state()
        p = ptm.amp_ph_damping_ptm(1.0, 0.42)

        assert np.allclose(p.dot(r), ground_state)


class TestGenAmpDamping:
    def test_equal_to_amp_damping(self):
        p1 = ptm.amp_ph_damping_ptm(gamma=0.42, lamda=0)
        p2 = ptm.gen_amp_damping_ptm(gamma_down=0.42, gamma_up=0)

        assert np.allclose(p1, p2)

    def test_strong_exciting_gives_excited_state(self):
        r = random_state()
        p = ptm.gen_amp_damping_ptm(gamma_down=0, gamma_up=1)

        assert np.allclose(p.dot(r), excited_state)


class TestKrausToPTM:

    def test_multiplicative_one_qubit(self):
        a = np.random.random((2, 2))
        b = np.random.random((2, 2))

        ptm_a = ptm.single_kraus_to_ptm(a)
        ptm_b = ptm.single_kraus_to_ptm(b)
        ptm_ab = ptm.single_kraus_to_ptm(np.matmul(a, b))
        assert np.allclose(ptm_ab, np.matmul(ptm_a, ptm_b))

    def test_multiplicative_two_qubit(self):
        a = np.random.random((4, 4))
        b = np.random.random((4, 4))

        ptm_a = ptm.double_kraus_to_ptm(a)
        ptm_b = ptm.double_kraus_to_ptm(b)
        ptm_ab = ptm.double_kraus_to_ptm(np.matmul(a, b))
        assert np.allclose(ptm_ab, np.matmul(ptm_a, ptm_b))
