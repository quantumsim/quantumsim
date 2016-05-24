import dm10
import numpy as np
import pytest



class TestDensityInit:
    def test_empty(self):
        dm = dm10.Density(10)
        assert dm._block_size == 32
        assert dm._grid_size == 32

    def test_numpy_array(self):
        n = 10
        a = np.zeros((2**10, 2**10))
        dm = dm10.Density(10, a)
        assert dm._block_size == 32
        assert dm._grid_size == 32

    def test_wrong_data(self):
        with pytest.raises(ValueError):
            dm = dm10.Density(10, "bla")


class TestDensityTrace:
    pass


    
