from collections.abc import Iterable
from .circuit import Circuit


def _toposort(trees):
    result = []
    all_used = set()

    while trees != []:
        trees.sort(key=lambda xy: len(all_used | xy[1]), reverse=True)
        smallest = trees.pop()
        all_used |= smallest[1]
        smallest = smallest[0]
        smallest.reverse()
        smallest = [x for n, x in smallest]

        new_trees = []
        for l, i in trees:
            l2 = [(n, x) for n, x in l if x not in smallest]
            new_trees.append((l2, i))

        trees = new_trees

        for s in smallest:
            if s not in result:
                result.append(s)

    return result


def partial_greedy_toposort(partial_order_set, target_set=None):
    """Given a list of partial orders `[p1, p2, ...]` of hashable items
    pi = [a_i0, a_i1, ...], representing the constraints

        a_i0 < a_i1 < ...

    construct a total ordering

        P: i0 < i1 < i2 < ...

    Some of the lists are denoted as target, and the ordering is chosen so
    that the number of overlaps between target lists is minimized, i.e.
    if p1 and p2 are targets, try to prevent

        a_10 < a_20 < a_1n < a_2n.

    This is done by a greedy algorithm.

    Parameters
    ----------

    partial_order: lists of lists of items
        Represents partial sequences targets: list of indices into
        partial_order, signifying which lists are targets
    targets: set
        The set of targets, which determine the overlaps that are minimized
        in the algortithm
    """
    if not isinstance(partial_order_set, Iterable):
        raise ValueError(
            "Partial expected as as iterable"
            "got {} instead".type(target_set))
    if not all([isinstance(order, Iterable) for order in partial_order_set]):
        raise ValueError(
            "Each element in partial order expected"
            "to be iterable")
    if target_set:
        if not isinstance(target_set, set):
            raise ValueError(
                "Targets expected as as iterable"
                "got {} instead".type(target_set))
    else:
        target_set = set()

    # drop out empty elements
    partial_order_set = [order for order in partial_order_set if order]

    order_dicts = [{i: j for i, j in zip(order[1:], order)}
                   for order in partial_order_set]

    trees = []
    for set_elem in partial_order_set:
        tree = []
        queue_stack = [(None, set_elem[-1])]
        while queue_stack:
            cur_ind, cur_elem = queue_stack.pop()
            tree.append((cur_ind, cur_elem))
            for dict_ind, order_dict in enumerate(order_dicts):
                next_elem = order_dict.get(cur_elem)
                if next_elem is not None:
                    queue_stack.append((dict_ind, next_elem))

        lists_used = {_ind for _ind, _elem in tree if _elem in target_set}
        trees.append((tree, lists_used))

    ordered_set = _toposort(trees)
    return ordered_set


def _reduces_bases(gate, qubit):
    qubit_ind = gate.qubits.index('Z')
    op = gate.operation_sympified()
    basis_out = op.bases_out[qubit_ind]
    return basis_out != basis_out.superbasis


def order(circuit):
    if not isinstance(circuit, Circuit):
        raise ValueError(
            "circuit expected to be an instance of"
            "quantumsim.cirucits.Circuit, got {} instead"
            .format(type(circuit)))

    sorted_gates = sorted(circuit.gates, key=lambda g: g.time_start)
    gate_dict = {ind: gate for ind, gate in enumerate(sorted_gates)}

    partial_order_inds = []
    target_inds = []

    for qubit_ind, qubit in enumerate(circuit.qubits):
        qubit_gates_dict = {ind: gate for ind, gate in gate_dict.items()
                            if qubit in gate.qubits}
        if any(_reduces_bases(gate) for gate in qubit_gates_dict.values()):
            target_inds.append(qubit_ind)
        partial_order_inds.append(qubit_gates_dict.keys())

    ordered_inds = partial_greedy_toposort(
        partial_order_inds, targets=target_inds)

    ordered_gates = [gate_dict[ind] for ind in ordered_inds]
    return Circuit(circuit.gates, ordered_gates)
