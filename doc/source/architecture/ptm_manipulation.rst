Pauli Transfer Matrix Manipulation
==================================

.. currentmodule:: quantumsim.operation

:class:`quantumsim.operation.Transformation` instances can be created using two
methods: :func:`Transformation.from_kraus` and :func:`Transformation.from_ptm`.
In order to get a transfer matrix, that will actually be used in
computations, :func:`Transformation.ptm` is used. It takes either Kraus, or
PTM provided during construction, and converts it to the computational
Pauli basis.

Before the usage, :class:`Transformation` instances should be compiled to
optimize input and output bases and reduce computational time and memory
consumption.

Conversion between Kraus and PTM representations
------------------------------------------------

