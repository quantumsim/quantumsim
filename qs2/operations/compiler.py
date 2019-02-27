from collections import deque
import numpy as np

from . import operation


class Node:
    def __init__(self, op, qubits):
        """

        Parameters
        ----------
        op : qs2.operations.Operation
        qubits : tuple of int
        """
        self.op = op
        self.qubits = qubits
        self.prev = {i: None for i in qubits}
        self.next = {i: None for i in qubits}
        self.bases_in_dict = {i: None for i in qubits}
        self.bases_out_dict = {i: None for i in qubits}
        self.merged = False

    def to_indexed_operation(self):
        assert self.op.bases_in == self.bases_in_tuple
        assert self.op.bases_out == self.bases_out_tuple
        return self.op.at(*self.qubits)

    @property
    def bases_in_tuple(self):
        return tuple(self.bases_in_dict[qubit] for qubit in self.qubits)

    @property
    def bases_out_tuple(self):
        return tuple(self.bases_out_dict[qubit] for qubit in self.qubits)


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

    def compile_next(self, optimize=True):
        node = self.get()
        b_in = node.bases_in_tuple
        b_out = tuple(bo or bi.superbasis for bo, bi in
                      zip(node.bases_out_tuple, node.bases_in_tuple))
        node.op = node.op.compile(b_in, b_out, optimize=optimize)

        for qubit, bi, bo in zip(node.qubits, node.op.bases_in,
                                 node.op.bases_out):
            node.bases_in_dict[qubit] = bi
            node.bases_out_dict[qubit] = bo
            if (node.prev[qubit] is not None and
                    node.prev[qubit].bases_out_dict[qubit] != bi):
                node.prev[qubit].bases_out_dict[qubit] = bi
                self.add(node.prev[qubit])
            if (node.next[qubit] is not None and
                    node.next[qubit].bases_in_dict[qubit] != bo):
                node.next[qubit].bases_in_dict[qubit] = bo
                self.add(node.next[qubit])


class CircuitGraph:
    def __init__(self, chain, bases_in, bases_out):
        self.starts = [None for _ in range(chain.num_qubits)]
        self.ends = [None for _ in range(chain.num_qubits)]
        self.nodes = []
        for op, qubtis in chain.operations:
            node_new = Node(op, qubtis)
            for qubit in qubtis:
                if self.starts[qubit] is None:
                    assert self.ends[qubit] is None
                    assert isinstance(qubit, int)
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
        return operation.Chain(*(node.to_indexed_operation()
                                 for node in self.nodes))

    def filter_merged(self):
        self.nodes = [node for node in self.nodes if not node.merged]

    def node_try_merge(self, node, where='next'):
        """

        Parameters
        ----------
        node: Node
        where: 'next' or 'prev'

        Returns
        -------

        """
        # Merge is possible, if there is only one next node
        # Assumes that bases are aligned
        if where == 'next':
            contr_candidates = set(node.next.values())
        elif where == 'prev':
            contr_candidates = set(node.prev.values())
        else:
            raise ValueError

        if len(contr_candidates) != 1 or None in contr_candidates:
            return
        other = contr_candidates.pop()

        d_self = len(node.qubits)
        d_other = len(other.qubits)
        i_other_out = list(range(d_other))
        i_other_in = list(range(d_other, 2 * d_other))

        contr_indices = [other.qubits.index(qubit)
                         for qubit in reversed(node.qubits)]
        if where == 'next':
            i_self_out = list(range(2 * d_other, 2 * d_other + d_self))
            i_self_in = [2 * d_other - i - 1 for i in contr_indices]
            for i, j in zip(contr_indices, i_self_out):
                i_other_in[d_other - i - 1] = j
        else:
            i_self_in = list(range(2 * d_other, 2 * d_other + d_self))
            i_self_out = [d_other - i - 1 for i in contr_indices]
            for i, j in zip(contr_indices, i_self_in):
                i_other_out[d_other - i - 1] = j

        try:
            other_ptm = np.einsum(node.op.ptm, i_self_out + i_self_in,
                                  other.op.ptm, i_other_out + i_other_in,
                                  optimize=True)
        except ValueError:
            einsum_args = [node.op.ptm, i_self_out + i_self_in,
                           other.op.ptm, i_other_out + i_other_in]
            raise


        if where == 'next':
            for qubit, node_prev in node.prev.items():
                other.prev[qubit] = node_prev
                assert other.bases_in_dict[qubit] == node.bases_out_dict[qubit]
                other.bases_in_dict[qubit] = node.bases_in_dict[qubit]
                other.op = operation.PTMOperation(
                    other_ptm, other.bases_in_tuple, other.bases_out_tuple)
                if node_prev is None:
                    self.starts[qubit] = other
                else:
                    node_prev.next[qubit] = other
        else:
            for qubit, node_next in node.next.items():
                other.next[qubit] = node_next
                assert other.bases_out_dict[qubit] == node.bases_in_dict[qubit]
                other.bases_out_dict[qubit] = node.bases_out_dict[qubit]
                other.op = operation.PTMOperation(
                    other_ptm, other.bases_in_tuple, other.bases_out_tuple)
                if node_next is None:
                    self.ends[qubit] = other
                else:
                    node_next.prev[qubit] = other

        node.merged = True


class ChainCompiler:
    """
    Parameters
    ----------
    chain : Chain
        A chain to compile
    """

    def __init__(self, chain, *, optimize=True):
        self.chain = chain
        self.optimize = optimize

    def compile(self, bases_in, bases_out):
        graph = CircuitGraph(self.chain, bases_in, bases_out)
        self.stage1_compile_all_nodes(graph, optimize=self.optimize)
        self.stage2_compress_chain(graph)
        return graph.to_operation()

    @staticmethod
    def stage1_compile_all_nodes(graph, optimize=True):
        queue = CompilerQueue(graph.nodes)
        while len(queue) > 0:
            queue.compile_next(optimize=optimize)

    @staticmethod
    def stage2_compress_chain(graph):
        """

        Parameters
        ----------
        graph : CircuitGraph

        Returns
        -------

        """
        for node in graph.nodes:
            graph.node_try_merge(node, 'next')
        graph.filter_merged()
        for node in reversed(graph.nodes):
            graph.node_try_merge(node, 'prev')
        graph.filter_merged()
