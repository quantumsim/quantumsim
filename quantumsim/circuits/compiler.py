from collections import deque

import numpy as np

from quantumsim.circuits import Gate, Circuit
from quantumsim.circuits.circuit import GateSetMixin


def optimize(circuit, bases_in=None, bases_out=None, qubits=None, *,
             optimizations=True, sv_cutoff=1e-5):
    """
    Returns an optimized for efficient computation version of a circuit.

    Parameters
    ----------
    circuit : quantumsim.Circuit
    bases_in : list of quantumsim.PauliBasis, optional
    bases_out : list of quantumsim.PauliBasis, optional
    qubits : list of hashable, optional
        List of qubit tags, used for ordering. Defaults to the order in `circuit`.
    optimizations : bool
        Whether to search possibility to reduce bases.
    sv_cutoff : float
        Minimal value of a singular value of Pauli transfer matrix, which is
        kept.

    Returns
    -------
    quantumsim.Circuit or quantumsim.Gate
    """
    if not isinstance(circuit, GateSetMixin):
        raise ValueError(f"`circuit` must be an instance of Circuit or Gate,"
                         f" got {type(circuit)}")
    graph = CircuitGraph(circuit, qubits or sorted(circuit.qubits), bases_in, bases_out)
    compile_graph(graph, optimizations=optimizations, sv_cutoff=sv_cutoff)
    return graph.to_circuit()


def compile_graph(graph, *, optimizations=True, sv_cutoff=1e-5):
    stage_align_bases(graph, optimizations=optimizations, sv_cutoff=sv_cutoff)
    stage_merge_nodes(graph)


def stage_align_bases(graph, *, optimizations=True, sv_cutoff=1e-5):
    queue = CompilerQueue(graph.nodes)
    while len(queue) > 0:
        compile_next_node_in_queue(queue, optimizations=optimizations,
                                   sv_cutoff=sv_cutoff)


def stage_merge_nodes(graph):
    """

    Parameters
    ----------
    graph : CircuitGraph

    Returns
    -------

    """
    for node in graph.nodes:
        try_merge_prev(graph, node)
    graph.filter_merged()
    for node in reversed(graph.nodes):
        try_merge_next(graph, node)
    graph.filter_merged()


def compile_next_node_in_queue(queue, *, optimizations=True, sv_cutoff=1e-5):
    """

    Parameters
    ----------
    queue : CompilerQueue
    optimizations : bool
    sv_cutoff : float
    """
    node = queue.get()
    b_in = node.bases_in_tuple
    b_out = tuple(bo or bi.superbasis for bo, bi in
                  zip(node.bases_out_tuple, node.bases_in_tuple))
    node.op = node.op.set_bases(b_in, b_out)
    if optimizations:
        b_in, b_out = optimal_bases(node, sv_cutoff=sv_cutoff)
        node.op = node.op.set_bases(b_in, b_out)

    for qubit, bi, bo in zip(node.qubit_indices, node.op.bases_in,
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

    if not node.is_arranged() and not node.is_placeholder:
        node.arrange()


def try_merge_next(graph, node):
    """

    Parameters
    ----------
    graph: CircuitGraph
    node: Node
    """
    # Merge is possible, if there is only one next node
    # Assumes that bases are aligned
    if node.is_placeholder:
        return
    contr_candidates = set(node.next.values())
    if len(contr_candidates) != 1 or None in contr_candidates:
        return
    other = contr_candidates.pop()
    if other.is_placeholder:
        return

    d_node = len(node.qubit_indices)
    d_other = len(other.qubit_indices)

    contr_indices = [other.qubit_indices.index(qubit)
                     for qubit in node.qubit_indices]
    other_out = list(range(d_other))
    other_in = list(range(d_other, 2 * d_other))
    node_out = list(range(2 * d_other, 2 * d_other + d_node))
    node_in = [other_in[i] for i in contr_indices]
    for i, j in zip(contr_indices, node_out):
        other_in[i] = j

    other_ptm = np.einsum(node.op.ptm, node_out + node_in,
                          other.op.ptm, other_out + other_in,
                          optimize='greedy')

    time_start = min(node.op.time_start, other.op.time_start)
    time_end = max(node.op.time_end, other.op.time_end)

    for qubit, node_prev in node.prev.items():
        other.prev[qubit] = node_prev
        other.bases_in_dict[qubit] = node.bases_in_dict[qubit]
        if node_prev is None:
            graph.starts[qubit] = other
        else:
            node_prev.next[qubit] = other

    other.op = Gate.from_ptm(other_ptm, other.bases_in_tuple, other.bases_out_tuple,
                             qubits=other.op.qubits, time_start=time_start,
                             duration=time_end-time_start)
    node.merged = True


def try_merge_prev(graph, node):
    """

    Parameters
    ----------
    graph: CircuitGraph
    node: Node
    """
    # Merge is possible, if there is only one previous node
    # Assumes that bases are aligned
    if node.is_placeholder:
        return
    contr_candidates = set(node.prev.values())
    if len(contr_candidates) != 1 or None in contr_candidates:
        return
    other = contr_candidates.pop()
    if other.is_placeholder:
        return

    d_node = len(node.qubit_indices)
    d_other = len(other.qubit_indices)

    contr_indices = [other.qubit_indices.index(qubit)
                     for qubit in node.qubit_indices]
    other_out = list(range(d_other))
    other_in = list(range(d_other, 2 * d_other))
    node_in = list(range(2 * d_other, 2 * d_other + d_node))
    node_out = contr_indices
    for i, j in zip(contr_indices, node_in):
        other_out[i] = j

    other_ptm = np.einsum(node.op.ptm, node_out + node_in,
                          other.op.ptm, other_out + other_in,
                          optimize='greedy')

    time_start = min(node.op.time_start, other.op.time_start)
    time_end = max(node.op.time_end, other.op.time_end)

    for qubit, node_next in node.next.items():
        other.next[qubit] = node_next
        other.bases_out_dict[qubit] = node.bases_out_dict[qubit]
        if node_next is None:
            graph.ends[qubit] = other
        else:
            node_next.prev[qubit] = other

    other.op = Gate.from_ptm(other_ptm, other.bases_in_tuple, other.bases_out_tuple,
                             qubits=other.op.qubits, time_start=time_start,
                             duration=time_end-time_start)
    node.merged = True


class Node:
    def __init__(self, op, qubit_indices):
        """

        Parameters
        ----------
        op : quantumsim.circuits.Gate
        qubit_indices : tuple of int
        """
        self.op = op
        # FIXME: Will fail for actual placeholders
        self.is_placeholder = op.ptm is None
        self.qubit_indices = list(qubit_indices)
        self.prev = {i: None for i in qubit_indices}
        self.next = {i: None for i in qubit_indices}
        self.bases_in_dict = {q: b for q, b in zip(qubit_indices, op.bases_in)}
        self.bases_out_dict = {q: b for q, b in zip(qubit_indices, op.bases_out)}
        self.merged = False

    @property
    def bases_in_tuple(self):
        return tuple(self.bases_in_dict[qubit] for qubit in self.qubit_indices)

    @property
    def bases_out_tuple(self):
        return tuple(self.bases_out_dict[qubit] for qubit in self.qubit_indices)

    def is_arranged(self):
        return np.all(self.qubit_indices[:-1] <= self.qubit_indices[1:])

    def arrange(self):
        order = np.argsort(np.argsort(self.qubit_indices))
        offset = len(order)
        idx = np.concatenate((order, order + offset))
        new_ptm = np.einsum(self.op.ptm, idx, sorted(idx))
        order = np.argsort(self.qubit_indices)
        self.qubit_indices = [self.qubit_indices[i] for i in order]
        qubits = [self.op.qubits[i] for i in order]
        self.op._qubits = qubits
        self.op.ptm = new_ptm
        self.op.bases_in = self.bases_in_tuple
        self.op.bases_out = self.bases_out_tuple
        self.op = Gate.from_ptm(new_ptm, self.bases_in_tuple, self.bases_out_tuple,
                                qubits=qubits, duration=self.op.duration,
                                time_start=self.op.time_start,
                                plot_metadata=self.op.plot_metadata,
                                repr_=self.op.repr)


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
    starts: list[Node]
    ends: list[Node]

    # noinspection PyTypeChecker
    def __init__(self, circuit, qubits, bases_in=None, bases_out=None):
        self.qubits = qubits
        if qubits:
            for q in circuit.qubits:
                if q not in qubits:
                    raise ValueError(f"Qubit \"{q}\" is not present in `qubits` list")
        self.starts = [None for _ in range(len(circuit.qubits))]
        self.ends = [None for _ in range(len(circuit.qubits))]
        self.nodes = []
        for i, op in enumerate(circuit.operations()):
            qubit_indices = [self.qubits.index(q) for q in op.qubits]
            node_new = Node(op, qubit_indices)
            for qb in qubit_indices:
                if self.starts[qb] is None:
                    self.starts[qb] = node_new
                    self.ends[qb] = node_new
                else:
                    old_end = self.ends[qb]
                    old_end.next[qb] = node_new
                    node_new.prev[qb] = old_end
                    self.ends[qb] = node_new
            self.nodes.append(node_new)
        if bases_in is not None:
            for qb, (b, node_start) in enumerate(zip(bases_in, self.starts)):
                node_start.bases_in_dict[qb] = b
        if bases_out is not None:
            for qb, (b, node_end) in enumerate(zip(bases_out, self.ends)):
                node_end.bases_out_dict[qb] = b

    def to_circuit(self):
        if len(self.nodes) > 1:
            gates = [node.op for node in self.nodes]
            return Circuit(gates)
        elif len(self.nodes) == 1:
            return self.nodes[0].op
        else:
            raise RuntimeError('No operations in the graph.')

    def filter_merged(self):
        self.nodes = [node for node in self.nodes if not node.merged]


def optimal_bases(node, *, sv_cutoff=1e-5):
    """Based on input or output bases provided, determine an optimal basis, throwing
    away all basis elements, that are guaranteed not to contribute to the result of PTM
    application.

    Circuits provide some restrictions on input and output basis. For example, after the
    ideal initialization gate system is guaranteed to stay in :math:`|0\rangle` state,
    which means that input basis will consist of a single element. Similarly, if after
    the gate application qubit will be measured, only :math:`|0\rangle` and
    :math:`|1\rangle` states need to be computed, therefore we may reduce output basis
    to the classical subbasis. This method is used to perform such sort of optimization:
    usage of subbasis instead of a full basis in a density matrix will exponentially
    reduce memory consumption and computational time.

    Parameters
    ----------
    node : Node
    sv_cutoff : float
        Singular values smaller than cutoff are discarded and treated as zeros.

    Returns
    -------
    opt_basis_in, opt_basis_out: tuple of quantumsim.bases.PauliBasis
        Subbases of input bases, that will contribute to computation.
    """
    if node.is_placeholder:
        return node.bases_in_tuple, node.bases_out_tuple

    d_in = np.prod([b.dim_pauli for b in node.op.bases_in])
    d_out = np.prod([b.dim_pauli for b in node.op.bases_out])
    u, s, vh = np.linalg.svd(node.op.ptm
                             .reshape(d_out, d_in), full_matrices=False)
    truncate_index = np.sum(s > sv_cutoff)

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
