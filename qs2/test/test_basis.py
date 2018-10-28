# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np

from qs2.basis import (
    gell_mann,
    general,
    PauliBasis,
    twolevel_0xy1,
    twolevel_ixyz)


class TestPauliBasis:

    def test_init(self):
        labels = ['test']
        vectors = np.array([[[1]]])
        b0 = PauliBasis(vectors, labels)
        assert len(b0.labels) == 1
        assert np.allclose(b0.vectors.shape,np.array([1,1,1]))


class TestBasis:

    def test_inits(self):
        b0 = gell_mann(2)
        b1 = general(2)
        assert len(b0.labels) == 4
        assert len(b1.labels) == 4
        assert np.allclose(b0.vectors.shape,np.array([4,2,2]))
        assert np.allclose(b1.vectors.shape,np.array([4,2,2]))
