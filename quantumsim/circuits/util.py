from collections.abc import Iterable
from .circuit import Circuit


def partial_greedy_toposort(partial_orders, targets=None):
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
    if targets:
        if not isinstance(targets, Iterable):
            raise ValueError(
                "Targets expected as as iterable"
                "got {} instead".type(targets))
        targets = set(targets)
    else:
        targets = set()

    # drop out empty lists
    partial_orders = [order for order in partial_orders if order]

    order_dicts = [{i: j for i, j in zip(order[1:], order)}
                   for order in partial_orders]

    trees = []
    for n, order in enumerate(partial_orders):
        tree = []
        to_do = [(None, order[-1])]
        while to_do:
            n, x = to_do.pop()
            tree.append((n, x))
            for n2, o2 in enumerate(order_dicts):
                x2 = o2.get(x)
                if x2 is not None:
                    to_do.append((n2, x2))

        lists_used = {n for n, x in tree if n in targets}
        trees.append((tree, lists_used))

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
