from ...models import Model
from ...circuits import Gate
from ...operations import qutrits as ops, ParametrizedOperation
from ... import bases
import numpy as np

DIM = 3
DEFAULT_DURATION = np.nan
basis = (bases.general(DIM),)
rad = np.deg2rad


def rotate(self, qubit):
    return Gate(qubit,
                DIM,
                ParametrizedOperation(
                    lambda phi, theta: ops.rotate_euler(rad(phi), rad(theta), -rad(phi)),
                    basis,
                ),
                DEFAULT_DURATION,
                plot_metadata={"style": "box", "label": "$R_{{{phi}}}({theta})$"})

def rotate_x(self, qubit):
    return Gate(qubit,
                DIM,
                ParametrizedOperation(lambda theta: ops.rotate_x(rad(theta)), basis),
                DEFAULT_DURATION,
                plot_metadata={"style": "box", "label": "$X({theta})$"})

def rotate_y(self, qubit):
    return Gate(qubit,
                DIM,
                ParametrizedOperation(lambda theta: ops.rotate_y(rad(theta)), basis),
                DEFAULT_DURATION,
                plot_metadata={"style": "box", "label": "$Y({theta})$"})