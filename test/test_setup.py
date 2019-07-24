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
                           match='Setup does not define "qubits" section'):
            Setup("gates: []")
        with pytest.raises(SetupLoadError,
                           match='Setup does not define "gates" section'):
            Setup("qubits: []")
        with pytest.raises(SetupLoadError, match="Default qubit parameters "
                                                 "defined repeatedly .*"):
            Setup("""
            qubits:
            - t1: 100
            - t1: 200
            gates: []
            """)
        with pytest.raises(SetupLoadError, match='Parameters for qubit "X" '
                                                 'defined repeatedly .*'):
            Setup("""
            qubits:
            - t1: 500
            - name: X
              t1: 100
            - name: X
              t1: 200
            gates: []
            """)
        with pytest.raises(SetupLoadError, match='.* include a field "name"'):
            Setup("""
            qubits: []
            gates:
            - param_name: value
            """)
        with pytest.raises(SetupLoadError,
                           match='.* either a string, or a list'):
            Setup("""
            qubits: []
            gates:
            - name: X
              qubits: 1
            """)
        Setup("""
            qubits: []
            gates:
            - name: X
              param_name: 1
            - name: X
              qubits: QUBIT
              param_name: 2
            """)
        with pytest.raises(SetupLoadError,
                           match='Default parameters for gate "X" are defined '
                                 'repeatedly in the setup'):
            Setup("""
            qubits: []
            gates:
            - name: X
              param_name: 1
            - name: X
              param_name: 2
            """)
        with pytest.raises(SetupLoadError,
                           match='Parameters for gate "X" on qubits QUBIT are '
                                 'defined repeatedly in the setup'):
            Setup("""
            qubits: []
            gates:
            - name: X
              qubits: QUBIT
              param_name: 1
            - name: X
              qubits: QUBIT
              param_name: 2
            """)

    def test_setup_get(self):
        setup = Setup("""
        qubits:
        - name: Q0
          t1: 100
        - name: Q1
          t1: 100
        - t1: 100000
        gates:
        - name: with_default
          qubits: Q0
          defined_param: 100
        - name: with_default
          defined_param: 0
        - name: without_default
          qubits: Q0
          defined_param: 100
        - name: twoqubit
          qubits: [ Q0, Q1 ]
          defined_param: 300
        """)
        assert setup.param_qubit('t1', 'Q0') == 100
        assert setup.param_qubit('t1', 'Q1') == 100
        assert setup.param_qubit('t1', 'Qother') == 100000
        with pytest.raises(KeyError, match='Parameter "t2" is not defined for'
                                           ' qubit "Q0"'):
            setup.param_qubit('t2', 'Q0')
        with pytest.raises(KeyError, match='Parameter "t1" is not defined for'
                                           ' qubit "Qother"'):
            Setup("""
            qubits:
            - name: Q0
              t1: 100
            - name: Q1
              t1: 100
            gates: []
            """).param_qubit('t1', 'Qother')

        assert setup.param_gate('defined_param', 'with_default', 'Q0') == 100
        assert setup.param_gate('defined_param', 'with_default', 'Q1') == 0
        assert setup.param_gate('defined_param', 'without_default', 'Q0') == 100
        with pytest.raises(KeyError, match='Parameter "defined_param" is not '
                                           'defined for gate "without_default" '
                                           'with qubits Q1'):
            setup.param_gate('defined_param', 'without_default', 'Q1')
        assert setup.param_gate('defined_param', 'twoqubit', 'Q0', 'Q1') == 300
        with pytest.raises(KeyError, match='Parameter "defined_param" is not '
                                           'defined for gate "twoqubit" '
                                           'with qubits Q1, Q0'):
            setup.param_gate('defined_param', 'twoqubit', 'Q1', 'Q0')
