from ...models import Model
from ...circuits import Gate
from ...operations import qutrits as ops, ParametrizedOperation
from ... import bases
import numpy as np

DIM = 3
basis = (bases.general(DIM),)


def rotate_euler(qubit):
    return Gate(qubit,
                DIM,
                ParametrizedOperation(
                    lambda phi, theta, lamda: ops.rotate_euler(phi, theta, lamda),
                    basis,
                ),
                duration=0,
                plot_metadata={"style": "box", "label": "$R({phi}, {theta}, {lamda})$"},
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


# TODO: would be nice to make display in the units of pi. Probably this requires
# to make plot_metadata accept functions