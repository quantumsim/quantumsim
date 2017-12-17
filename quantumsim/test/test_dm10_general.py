import quantumsim.ptm as ptm

import pytest
import numpy as np

from pytest import approx

try:
    import quantumsim.dm10_general as dm10g
except ImportError:
    pytest.skip("pycuda not installed, skip", allow_module_level=True)



def test_make_trivial():

    dm = dm10g.DensityGeneral([])

    assert dm.data.get() == approx(1)

    d = dm.get_diag()

    assert d == approx(1)


def test_get_diag_simple():
    pb = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([pb, pb])
    d = dm.get_diag()
    t = dm.trace()
    assert d == approx([1, 0, 0, 0])
    assert t == approx(1)

def test_get_diag_with_subbasis():

    pb = ptm.PauliBasis_0xy1()

    dm = dm10g.DensityGeneral([pb.get_subbasis([0]), pb])



    assert dm.data.shape == (1, 4)

    d = dm.get_diag()

    t = dm.trace()

    assert d == approx([1, 0])

    assert t == approx(1)

def test_cache_ptm():
    pb = ptm.PauliBasis_0xy1()

    dm = dm10g.DensityGeneral([pb, pb])

    dm._cached_gpuarray(np.array([1.0, 0.0, 1.0, 0.0]))

def test_simple_hadamard():
    pb = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([pb, pb])
    p = ptm.hadamard_ptm()

    dm.apply_ptm(0, p)

    d = dm.get_diag()
    t = dm.trace()
    pt0 = dm.partial_trace(0)
    pt1 = dm.partial_trace(1)

    assert pt0 == approx([0.5, 0.5])
    assert pt1 == approx([1, 0])
    assert t == approx(1)
    assert d == approx([0.5, 0, 0.5, 0])

    dm.apply_ptm(1, p)

    d = dm.get_diag()
    t = dm.trace()

    assert t == approx(1)
    assert d == approx([0.25, 0.25, 0.25, 0.25])

    dm.project_measurement(0, 1)
    
    assert dm.data.shape == (1, 4)

    d = dm.get_diag()
    t = dm.trace()

    assert t == approx(0.5)
    assert d == approx([0.25, 0.25])

    dm.project_measurement(1, 0)
    
    assert dm.data.shape == (1, 1)

    d = dm.get_diag()
    t = dm.trace()

    assert t == approx(0.25)
    assert d == approx([0.25])


def test_add_qubit():
    pb = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([pb])

    dm.add_ancilla(pb, 1)

    assert len(dm.bases) == 2
    assert dm.data.shape == (4,4)

    d = dm.get_diag()
    t = dm.trace()

    # atm, new ancillas are added as new msb

    assert d == approx([0, 0, 1, 0])
    assert t == approx(1)

def test_order():
    pb = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([pb])

    p = ptm.RotateXPTM(np.pi).get_matrix(pb)
    dm.apply_ptm(0, p)

    d = dm.get_diag()
    assert d == approx([0, 1])

    dm.add_ancilla(pb, 0)

    d = dm.get_diag()
    assert d == approx([0, 1, 0, 0])

    dm.apply_ptm(1, p)

    d = dm.get_diag()
    assert d == approx([1, 0, 0, 0])



def test_project():
    pb = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([pb, pb])

    p = ptm.hadamard_ptm()

    dm.apply_ptm(0, p)

    d1 = dm.data.get().ravel()
    dm.add_ancilla(pb, 1)
    dm.project_measurement(0, 1)

    d2 = dm.data.get().ravel()

    assert np.allclose(d1, d2)

def test_add_to_empty():
    pb = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([])
    dm.add_ancilla(pb, 1)
    assert dm.get_diag() == approx([0, 1])

def test_basis_transform_via_single_ptm():
    pb = ptm.PauliBasis_0xy1()
    pb2 = ptm.PauliBasis_ixyz()

    p = ptm.ProductPTM([]).get_matrix(pb, pb2)

    dm = dm10g.DensityGeneral([pb])

    assert dm.data.get() == approx([1, 0, 0, 0])

    dm.apply_ptm(0, p, new_basis=pb2)

    assert dm.bases[0] == pb2
    assert dm.data.get() == approx([np.sqrt(.5), 0, 0, np.sqrt(.5)])

def test_expand_ancilla_via_single_ptm():
    pb2 = ptm.PauliBasis_0xy1()
    pb = pb2.get_subbasis([0])

    p = ptm.ProductPTM([]).get_matrix(pb, pb2)
    print(p)

    dm = dm10g.DensityGeneral([pb])

    assert dm.data.get() == approx([1])

    dm.apply_ptm(0, p, new_basis=pb2)

    assert dm.bases[0] == pb2
    assert dm.data.get() == approx([1, 0, 0, 0])



def test_expand_and_rotate_ancilla_via_single_ptm():
    pb1 = ptm.PauliBasis_0xy1()
    pb = pb1.get_subbasis([0])

    pb2 = ptm.PauliBasis_ixyz()

    p = ptm.RotateXPTM(np.pi).get_matrix(pb, pb2)
    print(p)

    dm = dm10g.DensityGeneral([pb])

    assert dm.data.get() == approx([1])

    dm.apply_ptm(0, p, new_basis=pb2)

    assert dm.bases[0] == pb2
    assert dm.data.get() == approx([np.sqrt(.5), 0, 0, -np.sqrt(.5)])


def test_get_diag_with_tainted_work():

    pb = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([pb, pb])

    assert dm.data.gpudata.size == 16*8
    assert dm._work_data.gpudata.size == 16*8

    dm._work_data.fill(42)

    assert dm.get_diag() ==  approx([1, 0, 0, 0])


def test_simple_two_qubit_cnot():
    pb = ptm.PauliBasis_0xy1()

    # msb is control bit
    u = [[1, 0, 0, 0],
         [0, 1, 0, 0],           
         [0, 0, 0, 1],                                                 
         [0, 0, 1, 0]] 
    u = np.array(u)

    p_cnot = ptm.double_kraus_to_ptm(u).reshape(4,4,4,4)
    
    dm = dm10g.DensityGeneral([pb, pb])

    assert dm.get_diag() == approx([1, 0, 0, 0])

    # bit0 is the msb in ptm
    dm.apply_two_ptm(1, 0, p_cnot) 

    # cnot on ground state does nothing
    assert dm.get_diag() == approx([1, 0, 0, 0])

    p_flip = ptm.RotateXPTM(np.pi).get_matrix(pb)
    dm.apply_ptm(1, p_flip)

    print(dm.data)
    print(dm.get_diag())

    #bit 0 in dm is msb
    assert dm.get_diag() == approx([0, 1, 0, 0])

    dm.apply_two_ptm(0, 1, p_cnot) 
    print(dm.data)
    print(dm.get_diag())
    assert dm.get_diag() == approx([0, 0, 0, 1])


def test_make_qutrit():
    p = ptm.GeneralBasis(3)
    dm = dm10g.DensityGeneral([p])

    d = dm.get_diag()

    assert d == approx([1, 0, 0])

def test_excite_qutrit_by_two_rotations():
    b = ptm.GeneralBasis(3)
    dm = dm10g.DensityGeneral([b])

    u01 = np.zeros((3, 3))
    u01[0, 1] = 1
    u01[1, 0] = 1
    u01[2, 2] = 1

    ptm01 = ptm.ConjunctionPTM(u01).get_matrix(b)

    u12 = np.zeros((3, 3))
    u12[0, 0] = 1
    u12[1, 2] = 1
    u12[2, 1] = 1

    ptm12 = ptm.ConjunctionPTM(u12).get_matrix(b)

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

def test_qubit_plus_qutrit():
    b = ptm.GeneralBasis(3)
    b2 = ptm.PauliBasis_0xy1()
    dm = dm10g.DensityGeneral([b, b2])

    u01 = np.zeros((3, 3))
    u01[0, 1] = 1
    u01[1, 0] = 1
    u01[2, 2] = 1

    ptm01 = ptm.ConjunctionPTM(u01).get_matrix(b)

    u12 = np.zeros((3, 3))
    u12[0, 0] = 1
    u12[1, 2] = 1
    u12[2, 1] = 1

    ptm12 = ptm.ConjunctionPTM(u12).get_matrix(b)

    # excite to second state
    dm.apply_ptm(0, ptm01)
    dm.apply_ptm(0, ptm12)

    assert dm.get_diag() == approx([0, 0, 0, 0, 1, 0])
    assert dm.trace() == approx(1)

    # and the other qubit

    px = ptm.RotateXPTM(np.pi).get_matrix(b2)
    dm.apply_ptm(1, px)

    # and down again
    dm.apply_ptm(0, ptm12)
    dm.apply_ptm(0, ptm01)

    assert dm.trace() == approx(1)
    assert dm.get_diag() == approx([0, 1, 0, 0, 0, 0])

    dm.project_measurement(0, 0)
    assert dm.trace() == approx(1)
    dm.project_measurement(1, 1)
    assert dm.trace() == approx(1)

