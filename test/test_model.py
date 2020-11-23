from numpy import pi

from quantumsim import Model, Setup
from quantumsim.circuits import FinalizedCircuit, Circuit
from quantumsim.models import WaitingGate

from quantumsim.models import perfect_qubits as lib


def test_create_untimed_model():
    class SampleModel(Model):
        dim = 2

        @Model.gate()
        def rotate_y(self, qubit):
            return lib.rotate_y(qubit)

        @Model.gate()
        def cz(self, qubit_static, qubit_fluxed):
            return lib.cphase(qubit_static, qubit_fluxed, angle=pi)

    sample_setup = Setup("""
    setup: []
    """)
    m = SampleModel(sample_setup)

    assert len(m.rotate_y('D0').free_parameters) == 1
    assert len(m.rotate_y('D0', angle=0.5*pi).free_parameters) == 0

    _ = m.rotate_y('D0', angle=0.5*pi) + m.cz('D0', 'D1') + \
        m.rotate_y('D0', angle=-0.5*pi)


def test_create_timed_model():
    class SampleModel(Model):
        dim = 2

        @Model.gate(duration=20)
        def rotate_y(self, qubit):
            return (self.wait(qubit, 10) +
                    lib.rotate_x(qubit) +
                    self.wait(qubit, 10))

        @Model.gate(duration='t_twoqubit')
        def cphase(self, qubit_static, qubit_fluxed):
            return (self.wait(qubit_static, 0.5*self.p('t_twoqubit')) +
                    self.wait(qubit_fluxed, 0.5*self.p('t_twoqubit')) +
                    # FIXME should be `..., angle=pi)`
                    lib.cphase(qubit_static, qubit_fluxed) +
                    self.wait(qubit_static, 0.5*self.p('t_twoqubit')) +
                    self.wait(qubit_fluxed, 0.5*self.p('t_twoqubit')))

        @Model.gate(duration=lambda qubit, setup: 600 if qubit == 'D0' else 400)
        def strange_duration_gate(self, qubit):
            return lib.rotate_y(qubit, angle=pi)

        def finalize(self, circuit, bases_in=None, qubits=None):
            # Filter out waiting gates
            # In real life this will be replacing them to idling operators
            gates = [g for g in circuit.operations() if not isinstance(g, WaitingGate)]
            return FinalizedCircuit(Circuit(gates), qubits=qubits, bases_in=bases_in)

    sample_setup = Setup("""
    setup:
    - t_twoqubit: 40
    """)

    m = SampleModel(sample_setup)
    cnot = (m.rotate_y('D0', angle=0.5*pi) +
            m.cphase('D0', 'D1') +
            m.rotate_y('D0', angle=-0.5*pi))
    assert len(list(cnot.operations())) == 11
    cnot = m.finalize(cnot)
    assert len(list(cnot.compiled_circuit.operations())) == 3
    assert len(list(cnot(angle=pi).compiled_circuit.operations())) == 1
