import quantumsim.ptm as ptm
import numpy as np

class TestPTMBasisConversion:
    def test_tm_convert_unity(self):
        assert np.allclose(ptm.to_0xy1_basis(np.eye(4)), np.eye(4))
        assert np.allclose(ptm.to_0xy1_basis(np.eye(3)), np.eye(4))

    def test_hadamard(self):

        s2 = np.sqrt(0.5)
        ptm_hadamard_0xy1_should_be = np.array(
                [[0.5, s2, 0, 0.5],
                 [s2, 0, 0, -s2],
                 [0,0,-1, 0],
                 [0.5, -s2, 0, 0.5]]
                )

        ptm3x3hadamard = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        ptm_hadamard = ptm.to_0xy1_basis(ptm3x3hadamard)
        assert np.allclose(ptm_hadamard_0xy1_should_be, ptm_hadamard)

        ptm4x3hadamard = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])
        ptm_hadamard = ptm.to_0xy1_basis(ptm4x3hadamard)
        assert np.allclose(ptm_hadamard_0xy1_should_be, ptm_hadamard)

        ptm4x4hadamard = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])
        ptm_hadamard = ptm.to_0xy1_basis(ptm4x4hadamard)
        assert np.allclose(ptm_hadamard_0xy1_should_be, ptm_hadamard)


