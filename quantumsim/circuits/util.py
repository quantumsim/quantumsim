from collections.abc import Iterable
from .circuit import Circuit


def _toposort(trees):
    """
    Topological sorting of the trees of partial orders

    Parameters
    ----------
    trees : list
        A list of tuples of the tree branches and the list of targets indicies within the branch.

    Returns
    -------
    list
        The sorted list of all branch indicies according to the topological sort.
    """
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
            "got {} instead".format(type(target_set)))
    if not all([isinstance(order, Iterable) for order in partial_order_set]):
        raise ValueError(
            "Each element in partial order expected"
            "to be iterable")
    if target_set:
        if not isinstance(target_set, set):
            raise ValueError(
                "Targets expected as as iterable"
                "got {} instead".format(type(target_set)))
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

        lists_used = {_ind for _ind, _ in tree if _ind in target_set}
        trees.append((tree, lists_used))

    ordered_set = _toposort(trees)
    return ordered_set


def _reduces_bases(gate, qubit):
    """
    Checks if the operation that the gate implements
    leads to a reduction in the Pauli dimensionality
    of the outgoing Pauli basis vectors on the specified qubit.


    Parameters
    ----------
    gate : quantumsim.circut.Gate
        The gate that is checked for dimension-reducing operations.
    qubit : str
        The qubit on which the basis is checked for reduction.

    Returns
    -------
    True if the operation reduces the Pauli size of the basis on that qubit else False
    """
    op = gate.operation_sympified()
    qubit_ind = gate.qubits.index(qubit)
    for unit_op, unit_inds in op.units():
        if qubit_ind in unit_inds:
            basis_out = unit_op.bases_out[unit_inds.index(qubit_ind)]
            if basis_out.dim_pauli != basis_out.superbasis.dim_pauli:
                return True
    return False


def order(circuit):
    """
    Reorders the gates defined in the circuit, such that they are applied in
    temporal order. If any freedom exists when choosing the order of
    commuting gates, the order is chosen such that gates leading to a reduction
    of the size of the pauli basis vector  are applied "as soon as possible".

    Parameters
    ----------
    circuit : quantumsim.circuits.Circuit
        An unfinalized quantumsim cirucit instance.

    Returns
    -------
    quantumsim.circuits.Circuit
        A circuit instance with the topologoically sorted gates via
        a greedy algroithm, which priortizes the application of
        basis-size reducing gates.
    """
    if not isinstance(circuit, Circuit):
        raise ValueError(
            "circuit expected to be an instance of"
            "quantumsim.cirucits.Circuit, got {} instead"
            .format(type(circuit)))

    sorted_gates = sorted(circuit.gates, key=lambda g: g.time_start)
    gate_dict = {ind: gate for ind, gate in enumerate(sorted_gates)}

    partial_order_inds = []
    target_inds = set()

    for qubit_ind, qubit in enumerate(circuit.qubits):
        _q_gates = {ind: gate for ind, gate in gate_dict.items()
                    if qubit in gate.qubits}
        if any(_reduces_bases(gate, qubit) for gate in _q_gates.values()):
            target_inds.add(qubit_ind)
        partial_order_inds.append(list(_q_gates.keys()))

    ordered_inds = partial_greedy_toposort(
        partial_order_inds, target_set=target_inds)

    ordered_gates = [gate_dict[ind] for ind in ordered_inds]
    return Circuit(circuit.qubits, ordered_gates)
