from collections import deque
from itertools import repeat
import numpy as np

from .operation import Operation


class Node:
    def __init__(self, op, qubits):
        """

        Parameters
        ----------
        op : quantumsim.operations.Operation
        qubits : tuple of int
        """
        self.op = op
        self.qubits = list(qubits)
        self.prev = {i: None for i in qubits}
        self.next = {i: None for i in qubits}
        if hasattr(op, 'bases_in'):
            self.bases_in_dict = {q: b for q, b in zip(qubits, op.bases_in)}
        else:
            self.bases_in_dict = {q: None for q in qubits}
        if hasattr(op, 'bases_out'):
            self.bases_out_dict = {q: b for q, b in zip(qubits, op.bases_out)}
        else:
            self.bases_out_dict = {q: None for q in qubits}
        self.merged = False

    def to_indexed_operation(self):
        return self.op.at(*self.qubits)

    @property
    def op_ptm(self):
        return self.op.ptm(self.op.bases_in, self.op.bases_out)

    @property
    def bases_in_tuple(self):
        return tuple(self.bases_in_dict[qubit] for qubit in self.qubits)

    @property
    def bases_out_tuple(self):
        return tuple(self.bases_out_dict[qubit] for qubit in self.qubits)

    def is_arranged(self):
        return np.all(self.qubits[:-1] <= self.qubits[1:])

    def arrange(self):
        offset = max(self.qubits) + 1
        idx = self.qubits + [q + offset for q in self.qubits]
        new_ptm = np.einsum(self.op.ptm(self.op.bases_in, self.op.bases_out),
                            idx, sorted(idx))
        self.qubits = sorted(self.qubits)
        self.op = Operation.from_ptm(
            new_ptm, self.bases_in_tuple, self.bases_out_tuple)


class CompilerQueue:
    def __init__(self, iterable=None):
        self._queue = deque([])
        if iterable:
            for item in iterable:
                self.add(item)

    def add(self, item):
        if item not in self._queue:
            self._queue.append(item)

    def get(self):
        return self._queue.popleft()

    def __len__(self):
        return len(self._queue)


class CircuitGraph:
    # noinspection PyTypeChecker
    def __init__(self, chain, bases_in=None, bases_out=None):
        self.starts = [None for _ in range(chain.num_qubits)]
        self.ends = [None for _ in range(chain.num_qubits)]
        self.nodes = []
        bases_in = bases_in or repeat(None)
        bases_out = bases_out or repeat(None)
        for op, qubtis in chain.operations:
            node_new = Node(op, qubtis)
            for qubit in qubtis:
                if self.starts[qubit] is None:
                    self.starts[qubit] = node_new
                    self.ends[qubit] = node_new
                else:
                    old_end = self.ends[qubit]
                    old_end.next[qubit] = node_new
                    node_new.prev[qubit] = old_end
                    self.ends[qubit] = node_new
            self.nodes.append(node_new)
        for qubit, (b, node_start) in enumerate(zip(bases_in, self.starts)):
            node_start.bases_in_dict[qubit] = b
        for qubit, (b, node_end) in enumerate(zip(bases_out, self.ends)):
            node_end.bases_out_dict[qubit] = b

    def to_operation(self):
        if len(self.nodes) > 1:
            return Operation.from_sequence(
                [node.to_indexed_operation() for node in self.nodes])
        elif len(self.nodes) == 1:
            return self.nodes[0].op
        else:
            raise RuntimeError('No operations in the graph.')

    def filter_merged(self):
        self.nodes = [node for node in self.nodes if not node.merged]


class ChainCompiler:
    """
    Parameters
    ----------
    chain : _Chain
        A chain to compile
    optimize : bool
        Whether to throw away inactive degrees of freedom or not.
    sv_cutoff : float
        During the Pauli transfer matrix optimizations, singular value
        decomposition of a transfer matrix is used to determine optimal
        computational basis. All singular values less than `sv_cutoff`
        are considered weakly contributed and neglected. This attribute
        should be set before any compilation of a circuit, otherwise default
        is used (1e-5).
    """

    def __init__(self, chain, *, optimize=True, sv_cutoff=1e-5):
        self.chain = chain
        self.optimize = optimize
        self.sv_cutoff = sv_cutoff

    def compile_next(self, queue):
        """

        Parameters
        ----------
        queue : CompilerQueue
        """
        node = queue.get()
        b_in = node.bases_in_tuple
        b_out = tuple(bo or bi.superbasis for bo, bi in
                      zip(node.bases_out_tuple, node.bases_in_tuple))
        node.op = node.op.set_bases(b_in, b_out)
        if self.optimize:
            b_in, b_out = self.optimal_bases(node)
            node.op = node.op.set_bases(b_in, b_out)

        for qubit, bi, bo in zip(node.qubits, node.op.bases_in,
                                 node.op.bases_out):
            node.bases_in_dict[qubit] = bi
            node.bases_out_dict[qubit] = bo
            if (node.prev[qubit] is not None and
                    node.prev[qubit].bases_out_dict[qubit] != bi):
                node.prev[qubit].bases_out_dict[qubit] = bi
                queue.add(node.prev[qubit])
            if (node.next[qubit] is not None and
                    node.next[qubit].bases_in_dict[qubit] != bo):
                node.next[qubit].bases_in_dict[qubit] = bo
                queue.add(node.next[qubit])

        if not node.is_arranged():
            node.arrange()

    def optimal_bases(self, node):
        """Based on input or output bases provided, determine an optimal
        basis, throwing away all basis elements, that are guaranteed not to
        contribute to the result of PTM application.

        Circuits provide some restrictions on input and output basis. For
        example, after the ideal initialization gate system is guaranteed to
        stay in :math:`|0\rangle` state, which means that input basis will
        consist of a single element. Similarly, if after the gate application
        qubit will be measured, only :math:`|0\rangle` and :math:`|1\rangle`
        states need to be computed, therefore we may reduce output basis to
        the classical subbasis. This method is used to perform such sort of
        optimization: usage of subbasis instead of a full basis in a density
        matrix will exponentially reduce memory consumption and computational
        time.

        Parameters
        ----------
        node : Node

        Returns
        -------
        opt_basis_in, opt_basis_out: tuple of quantumsim.bases.PauliBasis
            Subbases of input bases, that will contribute to computation.
        """
        d_in = np.prod([b.dim_pauli for b in node.op.bases_in])
        d_out = np.prod([b.dim_pauli for b in node.op.bases_out])
        u, s, vh = np.linalg.svd(node.op_ptm
                                 .reshape(d_out, d_in), full_matrices=False)
        (truncate_index,) = (s > self.sv_cutoff).shape

        mask_in = np.any(
            np.abs(vh[:truncate_index]) > 1e-13, axis=0) \
            .reshape(tuple(b.dim_pauli for b in node.op.bases_in)) \
            .nonzero()
        mask_out = np.any(
            np.abs(u[:, :truncate_index]) > 1e-13, axis=1) \
            .reshape(tuple(b.dim_pauli for b in node.op.bases_out)) \
            .nonzero()

        opt_bases_in = []
        opt_bases_out = []
        for opt_bases, bases, mask in (
                (opt_bases_in, node.op.bases_in, mask_in),
                (opt_bases_out, node.op.bases_out, mask_out)):
            for basis, involved_indices in zip(bases, mask):
                # Figure out what single-qubit basis elements are not
                # involved at all
                unique_indices = np.unique(involved_indices)
                if len(unique_indices) < basis.dim_pauli:
                    # We can safely use a subbasis
                    opt_bases.append(basis.subbasis(unique_indices))
                else:
                    # Nothing can be thrown out
                    opt_bases.append(basis)

        return tuple(opt_bases_in), tuple(opt_bases_out)

    @staticmethod
    def try_merge_next(graph, node):
        """

        Parameters
        ----------
        graph: CircuitGraph
        node: Node

        Returns
        -------

        """
        # Merge is possible, if there is only one next node
        # Assumes that bases are aligned
        contr_candidates = set(node.next.values())
        if len(contr_candidates) != 1 or None in contr_candidates:
            return
        other = contr_candidates.pop()

        d_node = len(node.qubits)
        d_other = len(other.qubits)

        contr_indices = [other.qubits.index(qubit)
                         for qubit in node.qubits]
        other_out = list(range(d_other))
        other_in = list(range(d_other, 2 * d_other))
        node_out = list(range(2 * d_other, 2 * d_other + d_node))
        node_in = [other_in[i] for i in contr_indices]
        for i, j in zip(contr_indices, node_out):
            other_in[i] = j

        other_ptm = np.einsum(node.op_ptm, node_out + node_in,
                              other.op_ptm, other_out + other_in,
                              optimize=True)

        for qubit, node_prev in node.prev.items():
            other.prev[qubit] = node_prev
            other.bases_in_dict[qubit] = node.bases_in_dict[qubit]
            if node_prev is None:
                graph.starts[qubit] = other
            else:
                node_prev.next[qubit] = other

        other.op = Operation.from_ptm(
            other_ptm, other.bases_in_tuple, other.bases_out_tuple)
        node.merged = True

    @staticmethod
    def try_merge_prev(graph, node):
        """

        Parameters
        ----------
        graph: CircuitGraph
        node: Node

        Returns
        -------

        """
        # Merge is possible, if there is only one previous node
        # Assumes that bases are aligned
        contr_candidates = set(node.prev.values())
        if len(contr_candidates) != 1 or None in contr_candidates:
            return
        other = contr_candidates.pop()

        d_node = len(node.qubits)
        d_other = len(other.qubits)

        contr_indices = [other.qubits.index(qubit)
                         for qubit in node.qubits]
        other_out = list(range(d_other))
        other_in = list(range(d_other, 2 * d_other))
        node_in = list(range(2 * d_other, 2 * d_other + d_node))
        node_out = contr_indices
        for i, j in zip(contr_indices, node_in):
            other_out[i] = j

        other_ptm = np.einsum(node.op_ptm, node_out + node_in,
                              other.op_ptm, other_out + other_in,
                              optimize=True)

        for qubit, node_next in node.next.items():
            other.next[qubit] = node_next
            assert other.bases_out_dict[qubit] == node.bases_in_dict[qubit]
            other.bases_out_dict[qubit] = node.bases_out_dict[qubit]
            if node_next is None:
                graph.ends[qubit] = other
            else:
                node_next.prev[qubit] = other

        other.op = Operation.from_ptm(
            other_ptm, other.bases_in_tuple, other.bases_out_tuple)
        node.merged = True

    def compile(self, bases_in=None, bases_out=None):
        graph = CircuitGraph(self.chain, bases_in, bases_out)
        self.stage1_compile_all_nodes(graph)
        self.stage2_compress_chain(graph)
        return graph.to_operation()

    def stage1_compile_all_nodes(self, graph):
        queue = CompilerQueue(graph.nodes)
        while len(queue) > 0:
            self.compile_next(queue)

    def stage2_compress_chain(self, graph):
        """

        Parameters
        ----------
        graph : CircuitGraph

        Returns
        -------

        """
        for node in graph.nodes:
            self.try_merge_next(graph, node)
        graph.filter_merged()
        for node in reversed(graph.nodes):
            self.try_merge_prev(graph, node)
        graph.filter_merged()
