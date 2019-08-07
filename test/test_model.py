import warnings

from numpy import pi
from pytest import approx

from quantumsim import Model, Setup, bases, Operation
from quantumsim.circuits import FinalizedCircuit
from quantumsim.models.model import WaitPlaceholder
from quantumsim.operations import ParametrizedOperation

import quantumsim.operations.qubits as lib


def test_create_untimed_model():
    basis = (bases.general(2),)

    class SampleModel(Model):
        dim = 2

        @Model.gate()
        def rotate_y(self, qubit):
            return (
                ParametrizedOperation(lib.rotate_y, basis),
            )

        @Model.gate()
        def cphase(self, qubit_static, qubit_fluxed):
            return (
                lib.cphase(pi).at(qubit_static, qubit_fluxed),
            )

        def finalize(self, circuit, bases_in=None):
            return FinalizedCircuit(circuit, bases_in=bases_in)

    sample_setup = Setup("""
    setup: []
    """)

    m = SampleModel(sample_setup)
    cnot = m.rotate_y('D0', angle=0.5*pi) + m.cphase('D0', 'D1') + \
           m.rotate_y('D0', angle=-0.5*pi)

    assert cnot.operation.ptm(basis*2, basis*2) == approx(
        Operation.from_sequence(
            lib.rotate_y(0.5*pi).at(0),
            lib.cphase(pi).at(0, 1),
            lib.rotate_y(-0.5*pi).at(0),
        ).ptm(basis*2, basis*2))


def test_create_timed_model():
    basis = (bases.general(2),)

    class SampleModel(Model):
        dim = 2

        @Model.gate(duration=20)
        def rotate_y(self, qubit):
            return (
                self.wait(qubit, 10),
                ParametrizedOperation(lib.rotate_y, basis),
                self.wait(qubit, 10),
            )

        @Model.gate(duration='t_twoqubit')
        def cphase(self, qubit_static, qubit_fluxed):
            return (
                self.wait(qubit_static, 0.5*self.p('t_twoqubit')),
                self.wait(qubit_fluxed, 0.5*self.p('t_twoqubit')),
                lib.cphase(pi).at(qubit_static, qubit_fluxed),
                self.wait(qubit_static, 0.5*self.p('t_twoqubit')),
                self.wait(qubit_fluxed, 0.5*self.p('t_twoqubit')),
            )

        @Model.gate(duration=lambda qubit, setup: 600 if qubit == 'D0' else 400)
        def strange_duration_gate(self, qubit):
            return lib.rotate_y(pi)

        def finalize(self, circuit, bases_in=None):
            out = super(SampleModel, self).finalize(circuit, bases_in)
            # Sample compilation: filter out waiting gates
            operations = [unit for unit in out.operation.units()
                          if not isinstance(unit.operation, WaitPlaceholder)]
            out.operation = Operation.from_sequence(operations).compile(
                bases_in=bases_in)
            return out

    sample_setup = Setup("""
    setup:
    - t_twoqubit: 40
    """)

    m = SampleModel(sample_setup)
    cnot = m.rotate_y('D0', angle=0.5*pi) + m.cphase('D0', 'D1') + \
           m.rotate_y('D0', angle=-0.5*pi)
    cnot = m.finalize(cnot)

    assert cnot.operation.ptm(basis*2, basis*2) == approx(
        Operation.from_sequence(
            lib.rotate_y(0.5*pi).at(0),
            lib.cphase(pi).at(0, 1),
            lib.rotate_y(-0.5*pi).at(0),
        ).ptm(basis*2, basis*2))

    gate1 = m.strange_duration_gate('D0')
    assert gate1.duration == 600
    gate2 = m.strange_duration_gate('D1')
    assert gate2.duration == 400
