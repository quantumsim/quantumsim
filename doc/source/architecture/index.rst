Quantumsim Architecture Overview
================================

Internal data representation of Quantumsim uses superoperator formalism
[1]_ [2]_. We store states as vectors in Hilbert-Schmidt space of dimension
:math:`d^2`, where :math:`d = \sum_{i=1}^{N} d_i` is a sum of dimensionalities
of all qudits in a system. This is done by expanding density matrix and quantum
operations in a basis of Hermitian matrices (see `Pauli bases`_). In this
representation, quantum state :math:`\rho` is a real-valued vector,
and quantum operation is a real-valued matrix.

Pauli Bases
-----------

Orthonormality condition
^^^^^^^^^^^^^^^^^^^^^^^^

We want to represent any arbitrary :math`d`-dimensional density matrix in a
form:

.. math::

    \hat{\rho} = \sum_{i=0}^N \rho_i \hat{P}_i,

where :math:`\rho_i` are real numbers.
Any Hermitian :math:`d \times d` matrix has :math:`d^2` free parameters,
therefore a full basis :math:`\left\{ \hat{P}_i \right\}` will have
:math:`N = d^2` elements.
We will refer to :math:`d` as Hilbert dimensionality (`dim_hilbert` in code),
and :math:`N` as Pauli dimensionality (`dim_pauli`).

We will call Pauli basis orthonormal, if it fulfills condition:

.. math::

    \text{tr} \left( \hat{P}_i \hat{P}_j \right) = \delta_{ij}.

If this is the case, inverse transformation has the form:

.. math::

    \rho_i = \text{tr} \left( \hat{\rho} \hat{P}_i \right).

We operate only in orthonormal bases in Quantumsim, an attempt to create
non-orthonormal basis will raise an exception.


Common bases
^^^^^^^^^^^^

The simplest example of a Pauli basis for :math:`2 \times 2` Hermitian matrices
is a basis, that consists of four matrices

.. math::

    \left\{ \hat{I}/\sqrt{2},\ \hat{\sigma}_x/\sqrt{2},\
    \hat{\sigma}_y/\sqrt{2},\ \hat{\sigma}_z/\sqrt{2} \right\},

unit matrix and three Pauli matrices, normalized by :math:`\sqrt{d}`, where
:math:`d = 2` is a number of dimensions.
We will refer to this basis as IXYZ basis. It can be generalized to the
arbitrary number of dimensions, if we replace Pauli matrices with generalized
Gell-Mann matrices [3]_. This basis can be constructed in Quantumsim with
:func:`qs2.bases.gell_mann`.

Another useful choice is formed by the following set of matrices:

.. math::

    \left\{
    \left(\hat{I} + \hat{\sigma}_z\right)/2,\ \hat{\sigma}_x/\sqrt{2},\
    \hat{\sigma}_y/\sqrt{2},\ \left(\hat{I} - \hat{\sigma}_z\right)/2
    \right\}

We will refer to this basis as 0XY1 basis. It has an advantage,
that probabilities of measuring 0 and 1 correspond in it to the coefficients
in front of first and last element of this basis, without the necessity to
compute trace explicitly. We can generalize this basis for arbitrary number of
dimensions :math:`d`: first we take :math:`d` matrices with 1 on a diagonal,
for example for :math:`d=3`:

.. math::

    \left\{
    \begin{pmatrix}
        1 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0
    \end{pmatrix},\
    \begin{pmatrix}
        0 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 0
    \end{pmatrix},\
    \begin{pmatrix}
        0 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 1
    \end{pmatrix},
    \ \cdots\right\},

and then :math:`d^2-d` of :math:`\hat{\sigma}_x`- and
:math:`\hat{\sigma}_y`-like matrices:

.. math::

    \left\{\cdots,\
    \begin{pmatrix}
        0 & 1 & 0 \\
        1 & 0 & 0 \\
        0 & 0 & 0
    \end{pmatrix},\
    \begin{pmatrix}
        0 & -i & 0 \\
        i & 0 & 0 \\
        0 & 0 & 0
    \end{pmatrix},\
    \begin{pmatrix}
        0 & 0 & 1 \\
        0 & 0 & 0 \\
        1 & 0 & 0
    \end{pmatrix},\\
    \begin{pmatrix}
        0 & 0 & -i \\
        0 & 0 & 0 \\
        i & 0 & 0
    \end{pmatrix},\
    \begin{pmatrix}
        0 & 0 & 0 \\
        0 & 0 & 1 \\
        0 & 1 & 0
    \end{pmatrix},\
    \begin{pmatrix}
        0 & 0 & 0 \\
        0 & 0 & -i \\
        0 & i & 0
    \end{pmatrix},
    \right\}.

This basis can be constructed in Quantumsim with :func:`qs2.bases.general`
and is used as a default basis in Quantumsim.

State representation
--------------------

Suppose we have a system of :math:`N` qubits.
Let us fix a separate basis :math:`\left\{ \hat{P}^{(n)} \right\}` for each
qubit.
Now, the density matrix can be represented as follows:

.. math::

    \hat{\rho} = \sum_{i_1,\ldots,i_N} \rho_{i_1,\ldots,i_N}
    \hat{P}_{i_1}^{(1)} \otimes \ldots \otimes \hat{P}_{i_N}^{(N)},

where the sum runs over all elements of this basis.
In the case of full basis :math:`i_n \in \left[ 1 \ldots d_n^2 \right]`, but
in general we do not limit ourselves to operating in full bases: if we know from
the circuit, that some basis element is not needed, we will try to throw it away
in the sake of memory and speed.




.. [1] A. Y. Kitaev, A. H. Shen, and M. N. Vyalyi, "Classical and Quantum
       Computation" (American Mathematical Society, 2002).

.. [2] D. Greenbaum, "Introduction to Quantum Gate Set tomography",
       `arXiv:1509.02921 <https://arxiv.org/abs/1509.02921>`_.

.. [3] https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

