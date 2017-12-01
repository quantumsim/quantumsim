import quantumsim.dm_general_np as dmg
import pytest
import numpy as np

from quantumsim import ptm

from pytest import approx


class TestDensitySimple:
    def test_make_qutrit(self):
        dm = dmg.DensityGeneralNP([3])

        diag = dm.get_diag()
        assert len(diag) == 3
        assert diag == approx(np.array([1, 0, 0]))

    def test_make_qutrit_from_data(self):

        data = np.zeros((3, 3))

        data[1, 1] = 1

        dm = dmg.DensityGeneralNP([3], data)

        diag = dm.get_diag()
        assert len(diag) == 3
        assert diag == approx(np.array([0, 1, 0]))

    def test_excite_qutrit_by_two_rotations(self):

        dm = dmg.DensityGeneralNP([3])

        u01 = np.zeros((3, 3))
        u01[0, 1] = 1
        u01[1, 0] = 1
        u01[2, 2] = 1

        ptm01 = ptm.single_kraus_to_ptm_general(u01)

        u12 = np.zeros((3, 3))
        u12[0, 0] = 1
        u12[1, 2] = 1
        u12[2, 1] = 1

        ptm12 = ptm.single_kraus_to_ptm_general(u12)

        # excite to second state
        dm.apply_ptm(0, ptm01)
        dm.apply_ptm(0, ptm12)

        diag = dm.get_diag()
        assert len(diag) == 3
        assert diag == approx(np.array([0, 0, 1]))

        # and down again
        dm.apply_ptm(0, ptm12)
        dm.apply_ptm(0, ptm01)

        diag = dm.get_diag()
        assert len(diag) == 3
        assert diag == approx(np.array([1, 0, 0]))
