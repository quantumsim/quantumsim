from ...models import Model
from ...circuits import Gate
from ...operations import qubits as ops, ParametrizedOperation
from ... import bases
import numpy as np

DIM = 2
basis = (bases.general(DIM),)


def rotate_euler(qubit):
    return Gate(qubit,
                DIM,
                ParametrizedOperation(
                    lambda phi, theta, lamda: ops.rotate_euler(phi, theta, lamda),
                    basis,
                ),
                duration=0,
                plot_metadata={"style": "box", "label": "$R_({phi}, {theta}, {lamda})$"},
                repr_="RotEuler({phi}, {theta}, {lamda})")

def rotate_x(qubit):
    return Gate(qubit,
                DIM,
                ParametrizedOperation(lambda theta: ops.rotate_x(theta), basis),
                duration=0,
                plot_metadata={"style": "box", "label": "$X({theta})$"},
                repr_="X({theta})")


def rotate_y(qubit):
    return Gate(qubit,
                DIM,
                ParametrizedOperation(lambda theta: ops.rotate_y(theta), basis),
                duration=0,
                plot_metadata={"style": "box", "label": "$Y({theta})$"},
                repr_="Y({theta})")


def cphase(qubit1, qubit2):
    return Gate([qubit1, qubit2],
                DIM,
                ParametrizedOperation(lambda angle: ops.cphase(angle), basis*2),
                duration=0,
                plot_metadata={
                    "style": "line",
                    "markers": [
                        {"style": "marker", "label": "o"},
                        {"style": "marker", "label": "o"},
                    ],
                },
                repr_="CPhase({angle})")