Lindblad equation
=================

Constructing quantum operations from realistic Markovian error models is
commonly done using *Lindblad equation*:

.. math::

    \frac{d\hat{\rho}}{dt} = - i \left[ H, \hat{\rho} \right] +
    \sum_j \left[
        L_j \hat{\rho} L_j^\dagger -
        \frac{1}{2}\left\{ L_j^\dagger L_j, \hat{\rho} \right\}
    \right],

where :math:`\left[ A, B \right] = AB - BA` is a commutator,
:math:`\left\{ A, B \right\} = AB + BA` is an anticommutator,
and :math:`\hbar = 1`.
*Lindblad operators* :math:`L_j` describe the interaction with environment and
therefore can be used to describe error model for the gate [1]_.
Note the difference in the jump operator definition, in [1]_ they are
multiplied by two compared to the convention we use in Quantumsim.

In order to make use of Lindblad equations, we need to solve the Lindblad
equation and obtain Kraus operators or Pauli transfer matrix in some basis.
Expanding density matrix in some arbitrary orthonormal Pauli basis and using
orthonormality condition, we get:

.. math::
    \dot{\rho}_k = \mathcal{L}_{ki} \rho_i,

where

.. math::

    \mathcal{L}_{ki} = \text{tr} \left(
        - i \left[\hat{H}, \hat{P}_i \right] \hat{P}_k
        + \sum_j \left( 2 \hat{L}_j^\vphantom{\dagger}
          \hat{P}_i^\vphantom{\dagger} \hat{L}_j^\dagger
          \hat{P}_k^\vphantom{\dagger}
        - \left\{ \hat{L}_j^\dagger \hat{L}_j^\vphantom{\dagger},
          \hat{P}_i^\vphantom{\dagger} \right\} \hat{P}_k
    \right) \right).

We call the matrix :math:`\mathcal{L}` *Pauli Liouville matrix*.
Formally we can write the solution of a time evolution as:

.. math::

    \vec{\rho}(t^\prime) =
    \exp\left(\mathcal{L}(t^\prime - t)\right) \vec{\rho}(t),

where :math:`\vec{\rho}` is a Pauli vector, therefore Pauli transfer matrix for
the operation, described by Lindblad equation and acting time T is just:

.. math::

    R(t) = \exp(\mathcal{L} t).

Typically the computation of matrix exponent is a quite complex operation, so
amount of its usages during the code execution should be minimized.

.. [1] Sec. 8.4 of Nielson, M. A., and I. L. Chuang.
       "Quantum Computation and Quantum Information"
       Cambridge University Press (2000).
