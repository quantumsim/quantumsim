import quantumsim.ptm as ptm
import numpy as np

import pytest

pytest


# some states in 0xy1 basis
ground_state = np.array([1, 0, 0, 0])
excited_state = np.array([0, 0, 0, 1])

rng = np.random.RandomState()


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


class TestRotations:
    def test_xyz(self):
        ptm_x = ptm.rotate_x_ptm(np.pi / 2)
        ptm_z = ptm.rotate_x_ptm(np.pi / 2)
        ptm_y = ptm.rotate_x_ptm(-np.pi / 2)

        ptm_xyz = np.dot(ptm_y, np.dot(ptm_z, ptm_x))

        assert np.allclose(ptm_xyz, ptm_z)

    def test_power(self):
        ptm_x = ptm.rotate_x_ptm(2 * np.pi / 7)
        ptm_x7 = np.linalg.matrix_power(ptm_x, 7)
        assert np.allclose(ptm_x7, np.eye(4))

        ptm_y = ptm.rotate_y_ptm(2 * np.pi / 7)
        ptm_y7 = np.linalg.matrix_power(ptm_y, 7)
        assert np.allclose(ptm_y7, np.eye(4))

        ptm_z = ptm.rotate_z_ptm(2 * np.pi / 7)
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

        state = ptm.rotate_x_ptm(np.pi / 2).dot(state)
        state = ptm.rotate_y_ptm(np.pi / 2).dot(state)
        state = ptm.rotate_x_ptm(-np.pi / 2).dot(state)

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


class TestPauliBasis:
    def test_basic(self):

        pb = ptm.PauliBasis_0xy1()

        assert len(pb.computational_basis_vectors) == 2
        assert pb.comp_basis_indices == dict(enumerate([0, 3]))
        assert pb.dim_hilbert == 2
        assert pb.dim_pauli == 4

        pb = ptm.PauliBasis_ixyz()

        assert len(pb.computational_basis_vectors) == 2
        assert pb.comp_basis_indices == dict(enumerate([None, None]))
        assert pb.dim_hilbert == 2
        assert pb.dim_pauli == 4

        pb = ptm.GeneralBasis(2)

        assert len(pb.computational_basis_vectors) == 2
        assert pb.comp_basis_indices == dict(enumerate([0, 1]))
        assert pb.dim_hilbert == 2
        assert pb.dim_pauli == 4

        pb = ptm.GeneralBasis(3)

        assert len(pb.computational_basis_vectors) == 3
        assert pb.comp_basis_indices == dict(enumerate([0, 1, 2]))
        assert pb.dim_hilbert == 3
        assert pb.dim_pauli == 9


some_pauli_bases = [
    ptm.PauliBasis_0xy1(),
    ptm.PauliBasis_ixyz(),
    ptm.GeneralBasis(2),
    ptm.GeneralBasis(3)
]

class TestBasis:
    @pytest.mark.parametrize("pb", some_pauli_bases)
    def test_orthonormal(self, pb):
        pb.check_orthonormality()

    @pytest.mark.skip(reason="not implemented yet")
    def test_singleton(self):
        pb = ptm.GeneralBasis(3)
        pb2 = ptm.GeneralBasis(3)
        assert pb is pb2


class TestPTMs:

    @pytest.mark.parametrize("pb", some_pauli_bases)
    def test_unit_matrix_conjunction(self, pb):

        e_hilbert = np.eye(pb.dim_hilbert)

        unit_ptm = ptm.ConjunctionPTM(e_hilbert)

        e_pauli = unit_ptm.get_matrix(pb)

        assert e_pauli == pytest.approx(np.eye(pb.dim_pauli))

    @pytest.mark.parametrize("pb", some_pauli_bases)
    def test_unit_matrix_lincomb(self, pb):

        e_hilbert = np.eye(pb.dim_hilbert)

        unit_ptm = ptm.ConjunctionPTM(e_hilbert)

        unit_ptm2 = 0.5 * unit_ptm + 0.5 * unit_ptm

        e_pauli = unit_ptm2.get_matrix(pb)
        assert e_pauli == pytest.approx(np.eye(pb.dim_pauli))
        assert len(unit_ptm2.elements) == 1

        unit_ptm2 *= -1
        unit_ptm2 -= 2 * unit_ptm2

        e_pauli = unit_ptm2.get_matrix(pb)
        assert e_pauli == pytest.approx(np.eye(pb.dim_pauli))
        assert len(unit_ptm2.elements) == 1

        unit_ptm2 = unit_ptm2 * 0.5 + 0.5 * unit_ptm

        e_pauli = unit_ptm2.get_matrix(pb)
        assert e_pauli == pytest.approx(np.eye(pb.dim_pauli))
        assert len(unit_ptm2.elements) == 1

    @pytest.mark.parametrize("pb", some_pauli_bases)
    def test_unit_product(self, pb):

        unit_ptm = ptm.ProductPTM([])
        e_pauli = unit_ptm.get_matrix(pb)
        assert e_pauli == pytest.approx(np.eye(pb.dim_pauli))

    def test_0xy1_basis_explicit_on_hadamard(self):
        s2 = np.sqrt(0.5)
        ptm_hadamard_0xy1_should_be = np.array(
            [[0.5, s2, 0, 0.5],
             [s2, 0, 0, -s2],
             [0, 0, -1, 0],
             [0.5, -s2, 0, 0.5]]
        )

        pb = ptm.PauliBasis_0xy1()
        hadamard = ptm.ConjunctionPTM([[s2, s2], [s2, -s2]])

        ptm_hadamard = hadamard.get_matrix(pb)
        assert np.allclose(ptm_hadamard_0xy1_should_be, ptm_hadamard)


some_2d_pauli_bases = [
    ptm.PauliBasis_0xy1(),
    ptm.PauliBasis_ixyz(),
    ptm.GeneralBasis(2),
]


class TestAmpPhaseDampingPTM:
    @pytest.mark.parametrize("pb", some_2d_pauli_bases)
    def test_does_nothing_to_ground_state(self, pb):
        ground_state = pb.computational_basis_vectors[0]
        p = ptm.AmplitudePhaseDampingPTM(0.23, 0.42).get_matrix(pb)

        assert np.allclose(p @ ground_state, ground_state)

    def test_strong_damping_gives_ground_state(self):
        r = random_state()  # is in 0xy1 basis
        pb = ptm.PauliBasis_0xy1()
        p = ptm.AmplitudePhaseDampingPTM(1.0, 0.42).get_matrix(pb)

        assert np.allclose(p.dot(r), ground_state)


class TestRotationsPTM:
    @pytest.mark.parametrize("pb", some_2d_pauli_bases)
    def test_product_of_one(self, pb):
        for Rot in [ptm.RotateXPTM, ptm.RotateYPTM, ptm.RotateZPTM]:
            ptm_x = Rot(np.pi / 2)

            # explicit construction
            ptm_x2 = ptm.ProductPTM([ptm_x])

            mat_x2 = ptm_x2.get_matrix(pb)
            mat_x = ptm_x.get_matrix(pb)

            assert np.allclose(mat_x2, mat_x)

    @pytest.mark.parametrize("pb", some_2d_pauli_bases)
    def test_xyz(self, pb):
        ptm_x = ptm.RotateXPTM(np.pi / 2)
        ptm_z = ptm.RotateXPTM(np.pi / 2)
        ptm_y = ptm.RotateXPTM(-np.pi / 2)

        # explicit construction
        ptm_xyz = ptm.ProductPTM([ptm_x, ptm_y, ptm_z])
        assert len(ptm_xyz.elements) == 3

        # implicit construction
        ptm_xyz2 = ptm_x @ ptm_y @ ptm_z
        assert len(ptm_xyz2.elements) == 3

        mat_xyz = ptm_xyz.get_matrix(pb)
        mat_xyz2 = ptm_xyz2.get_matrix(pb)
        mat_z = ptm_z.get_matrix(pb)

        assert np.allclose(mat_xyz, mat_z)
        assert np.allclose(mat_xyz2, mat_z)

    @pytest.mark.parametrize("pb", some_2d_pauli_bases)
    @pytest.mark.skip()
    def test_xyz2(self, pb):
        # test handedness
        ptm_x = ptm.RotateXPTM(np.pi / 2)
        ptm_z = ptm.RotateZPTM(np.pi / 2)
        ptm_y = ptm.RotateYPTM(-np.pi / 2)

        #TODO

    @pytest.mark.parametrize("pb", some_2d_pauli_bases)
    def test_power(self, pb):
        for Rot in [ptm.RotateXPTM, ptm.RotateYPTM, ptm.RotateZPTM]:
            ptm_x = Rot(2 * np.pi / 7)
            ptm_x7 = ptm.ProductPTM([ptm_x] * 7)
            mat_x7 = ptm_x7.get_matrix(pb)
            assert np.allclose(mat_x7, np.eye(4))

    @pytest.mark.parametrize("pb", some_2d_pauli_bases)
    def test_excite_deexcite_ground_state(self, pb):

        ptm_x = ptm.RotateXPTM(np.pi).get_matrix(pb)
        ptm_y = ptm.RotateYPTM(np.pi).get_matrix(pb)
        ptm_z = ptm.RotateZPTM(np.pi).get_matrix(pb)

        ground_state = pb.computational_basis_vectors[0]
        state = ground_state
        state = np.dot(ptm_x, state)
        state = np.dot(ptm_y, state)
        state = np.dot(ptm_z, state)
        state = np.dot(ptm_x, state)
        state = np.dot(ptm_y, state)
        state = np.dot(ptm_z, state)

        assert np.allclose(state, ground_state)

    @pytest.mark.parametrize("pb", some_2d_pauli_bases)
    def test_xyx_ground_state(self, pb):

        ground_state = pb.computational_basis_vectors[0]
        state = ground_state

        state = ptm.RotateXPTM(np.pi / 2).get_matrix(pb) @ state
        state = ptm.RotateYPTM(np.pi / 2).get_matrix(pb) @ state
        state = ptm.RotateXPTM(-np.pi / 2).get_matrix(pb) @ state

        assert np.allclose(state, ground_state)
