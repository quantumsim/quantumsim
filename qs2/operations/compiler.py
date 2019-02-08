from collections import deque
import numpy as np

from . import operation


class Node:
    def __init__(self, operation, qubits):
        self.op = operation
        self.qubits = qubits
        self.prev = {i: None for i in qubits}
        self.next = {i: None for i in qubits}
        self.ptm = None
        self.bases_in = {i: None for i in qubits}
        self.bases_out = {i: None for i in qubits}
        self.merged = False

    def to_indexed_operation(self):
        return operation.Transformation.from_ptm(
            self.ptm,
            tuple((self.bases_in[qubit] for qubit in self.qubits)),
            tuple((self.bases_out[qubit] for qubit in self.qubits))
        ).at(*self.qubits)


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

    def compile_next(self):
        node = self.get()
        b_in = tuple((node.bases_in[q] for q in node.qubits))
        b_out = tuple((node.bases_out[qubit] or node.bases_in[qubit].superbasis
                       for qubit in node.qubits))
        opt_b_in, opt_b_out = node.op.optimal_bases(b_in, b_out)
        node.ptm = node.op.ptm(opt_b_in, opt_b_out)
        assert node.ptm is not None

        for qubit, bi, bo in zip(node.qubits, opt_b_in, opt_b_out):
            node.bases_in[qubit] = bi
            node.bases_out[qubit] = bo
            if node.prev[qubit] is not None and node.prev[qubit].bases_out[qubit] != bi:
                node.prev[qubit].bases_out[qubit] = bi
                self.add(node.prev[qubit])
            if node.next[qubit] is not None and node.next[qubit].bases_in[qubit] != bo:
                node.next[qubit].bases_in[qubit] = bo
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
            node_start.bases_in[qubit] = b
        for qubit, (b, node_end) in enumerate(zip(bases_out, self.ends)):
            node_end.bases_out[qubit] = b

    def to_chain(self):
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
        contr_indices = [other.qubits.index(qubit) for qubit in node.qubits]

        i_other_out = list(range(d_other))
        i_other_in = list(range(d_other, 2*d_other))

        if where == 'next':
            i_self_out = list(range(2*d_other, 2*d_other+d_self))
            i_self_in = [i_other_in[i] for i in contr_indices]
            for i, j in zip(contr_indices, i_self_out):
                i_other_in[i] = j
        else:
            i_self_in = list(range(2*d_other, 2*d_other+d_self))
            i_self_out = [i_other_out[i] for i in contr_indices]
            for i, j in zip(contr_indices, i_self_in):
                i_other_out[i] = j

        other.ptm = np.einsum(node.ptm, i_self_out + i_self_in,
                              other.ptm, i_other_out + i_other_in,
                              optimize=True)

        if where == 'next':
            for qubit, node_prev in node.prev.items():
                other.prev[qubit] = node_prev
                assert other.bases_in[qubit] == node.bases_out[qubit]
                other.bases_in[qubit] = node.bases_in[qubit]
                if node_prev is None:
                    self.starts[qubit] = other
                else:
                    node_prev.next[qubit] = other
        else:
            for qubit, node_next in node.next.items():
                other.next[qubit] = node_next
                assert other.bases_out[qubit] == node.bases_in[qubit]
                other.bases_out[qubit] = node.bases_out[qubit]
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
    def __init__(self, chain):
        self.chain = chain

    def compile(self, bases_in, bases_out):
        graph = CircuitGraph(self.chain, bases_in, bases_out)
        self.stage1_compile_all_nodes(graph)
        self.stage2_compress_chain(graph)
        return graph.to_chain()

    @staticmethod
    def stage1_compile_all_nodes(graph):
        queue = CompilerQueue(graph.nodes)
        while len(queue) > 0:
            queue.compile_next()

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
