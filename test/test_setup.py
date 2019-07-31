import os

import pytest

from quantumsim import Setup
from quantumsim.setups.setup import SetupLoadError

test_dir = os.path.dirname(os.path.abspath(__file__))


class TestSetup:
    def test_setup_load(self):
        _ = Setup.from_file(os.path.join(test_dir,
                            'time_aware_setup_with_defaults.yaml'))

        with pytest.raises(SetupLoadError,
                           match='Unknown setup schema version: 10'):
            Setup("version: 10")
        with pytest.raises(SetupLoadError,
                           match='Setup does not define "setup" section'):
            Setup("name: Broken setup")
        with pytest.raises(SetupLoadError, match="Default qubit parameters "
                                                 "defined repeatedly .*"):
            Setup("""
            setup:
            - t1: 100
            - t1: 200
            """)
        with pytest.raises(SetupLoadError, match='Parameters for qubit "X" '
                                                 'defined repeatedly .*'):
            Setup("""
            setup:
            - t1: 500
            - qubit: X
              t1: 100
            - qubit: X
              t1: 200
            """)
        Setup("""
            setup:
            - param_name: 1
            - qubit: X
              param_name: 2
            """)

    def test_setup_get(self):
        setup = Setup("""
        setup:
        - qubit: Q0
          param_with_default: 2
          t1: 100
        - qubits: [ Q1, Q2 ]
          t1: 100
          param_without_default: 10
        - param_with_default: 1
        """)
        assert setup.param('param_with_default', 'Q0') == 2
        assert setup.param('param_with_default', 'Q1') == 1
        assert setup.param('param_with_default', 'Q5', 'Q10') == 1
        assert setup.param('param_without_default', 'Q1', 'Q2') == 10
        with pytest.raises(KeyError, match=r'Parameter "param_without_default" '
                                           r'is not defined for qubit\(s\) Q1'):
            setup.param('param_without_default', 'Q1')
        with pytest.raises(KeyError,
                           match=r'Parameter "param_without_default" is not '
                                 r'defined for qubit\(s\) Q1, Q0'):
            setup.param('param_without_default', 'Q1', 'Q0')
        with pytest.raises(KeyError, match=r'Parameter "t1" is not defined for'
                                           r' qubit\(s\) Qother'):
            Setup("""
            setup:
            - qubit: Q0
              t1: 100
            - qubit: Q1
              t1: 100
            """).param('t1', 'Qother')
