import quantumsim.dm10_general as dm10g
import quantumsim.ptm as ptm

import pycuda.tools

import pytest
import numpy as np

from pytest import approx


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
