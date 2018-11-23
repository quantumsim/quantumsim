Quantumsim Architecture Overview
================================

Internal data representation of Quantumsim uses superoperator formalism
[1]_ [2]_. We store states as vectors in Hilbert-Schmidt space of dimension
:math:`d^2`, where :math:`d = \sum_{i=1}^{N} d_i` is a sum of dimensionalities
of all qudits in a system. This is done by expanding density matrix and quantum
operations in a basis of Hermitean matrices (see `Pauli bases`_). In this
representation, quantum state :math:`\rho` is a real-valued vector,
and quantum operation is a real-valued matrix.

Pauli Bases
-----------

IXYZ basis
^^^^^^^^^^

The simpilest example of a Pauli basis is a basis, that consists of four
matrices :math:`\left\{ \hat{I},\ \hat{\sigma}_x,\ \hat{\sigma}_y,\
\hat{\sigma}_z \right\}` of unit matrices and three Pauli matrices.
This is a full basis in a vector space of :math:`2 \times 2` complex Hermitian
matrices. Therefore, a state of a two-level system with a density matrix
:math:`\hat{\rho}` can be represented as a vector
:math:`\left| \rho \right\rangle` with components :math:`\rho_i` as follows:

.. math::

    \hat{\rho} = \sum_{i=0}^3 \rho_i \hat{P}_i,

where :math:`\hat{P}_i` are the elements of the basis above. Coefficients of
this vector can be determined as:

.. math::

    \rho_i = \frac12 \text{Tr} \hat{P}_i \hat{\rho}

We will refer to this basis as IXYZ basis. It can be generalized to the
arbitrary number of dimensions, if we replace Pauli matrices with generalized
Gell-Mann matrices [3]_. This basis can be constructed in Quantumsim with
:func:`qs2.bases.gell_mann`.

0XY1 basis
^^^^^^^^^^

Another useful example of a Pauli basis is formed by the following set of
matrices:

.. math::

    \left\{
    \frac{\hat{I} + \hat{\sigma}_z}{2}, \
    \hat{\sigma}_x,\
    \hat{\sigma}_y,\
    \frac{\hat{I} - \hat{\sigma}_z}{2}
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

.. [1] A. Y. Kitaev, A. H. Shen, and M. N. Vyalyi, "Classical and Quantum
       Computation" (American Mathematical Society, 2002).

.. [2] D. Greenbaum, "Introduction to Quantum Gate Set tomography",
       `arXiv:1509.02921 <https://arxiv.org/abs/1509.02921>`_.

.. [3] https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

