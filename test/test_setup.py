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

