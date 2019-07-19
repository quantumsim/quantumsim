# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt


def partial_greedy_toposort(partial_orders, targets=set()):
    """Given a list of partial orders ``[p1, p2, ...]`` of hashable items
    ``pi = [a_i0, a_i1, ...]``, representing the constraints

    .. math::

        a_{i0} < a_{i1} < ...

    construct a total ordering :math:`P: i_0 < i_1 < i_2 < ...`

    Some of the lists are denoted as target,
    and the ordering is chosen so that the number of overlaps between target
    lists is minimized, i.e. if ``p1`` and ``p2`` are targets, try to prevent

    .. math::

        a_{10} < a_{20} < a_{1n} < a_{2n}.

    This is done by a greedy algorithm.

    Parameters
    ----------
    partial_order: list
        List of lists of items, representing partial sequences
    targets: list
        List of indices into partial_order, signifying which lists are targets
    """

    targets = set(targets)

    # drop out empty lists
    partial_orders = [po for po in partial_orders if po]

    order_dicts = []
    for n, p in enumerate(partial_orders):
        order_dict = {i: j for i, j in zip(p[1:], p)}
        order_dicts.append(order_dict)

    trees = []
    for n, p in enumerate(partial_orders):
        tree = []
        to_do = [(None, p[-1])]
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


def simple_toposort(partial_orders, targets):
    """A toposort that assumes that each qubit is only measured
    once during the circuit, and that measured qubits will sit
    outside the density matrix (whilst others will not).

    This means that we can solve a slightly simpler problem;
    find the order such that ancilla qubits are added as late
    as possible.

    Parameters
    ------

    partial_orders : list of lists of integers
        list of gates attached to each qubit
    targets : dict of ints
        list of measurement gates to be targetted.
        targets[n] corresponds to the corresponding gate
        index for the target measurement of qubit n.
    """

    MAX_COUNT=1000

    # For each qubit, get the gates that need to be performed
    # before measurement.
    used_gates = {
        q: [g for g in partial_orders[q] if g < targets[q]]
        for q in targets}

    # Links contains any gates shared between two qubits.
    links = {
        q1: {q2: [g for g in po if g in po2]
             for q2, po2 in enumerate(partial_orders) if q1 != q2}
        for q1, po in enumerate(partial_orders)}

    # for each target, qubit_partial_order contains the list
    # of qubits that must be inserted into the dm before this
    # qubit is measured.
    qubit_partial_order = {q: {q2 for q2 in links[q] if
                               len(links[q][q2]) > 0 and
                               min(links[q][q2]) < targets[q]}
                           for q in targets}
    for q in targets:
        for count in range(MAX_COUNT):
            new_qubits = {
                q3 for q2 in qubit_partial_order[q]
                for q3 in links[q2] if
                links[q2][q3] and links[q2][q] and
                min(links[q2][q3]) < max(links[q2][q])}
            if new_qubits.issubset(qubit_partial_order[q]):
                break
            qubit_partial_order[q] |= new_qubits
            if count >= MAX_COUNT-2:
                raise ValueError('caught inf loop')

    # Make the total order in which the target qubits should be
    # added to the circuit
    total_order = []
    while True:
        new_qubits = [
            q1 for q1 in targets if q1 not in total_order and len([
                q2 for q2 in targets if q2 in qubit_partial_order[q1]
                and q2 not in total_order]) == 0]
        if len(new_qubits) == 0:
            if len(total_order) != len(targets):
                raise ValueError('There appears to be a loop')
            break
        total_order += new_qubits

    ordered_gates = []
    for q in total_order:
        add_to_gate(
            q, targets[q], links, ordered_gates, partial_orders)

    # Add other gates
    unused_gates = sorted([
        g for po in partial_orders for g in po
        if g not in ordered_gates])

    return ordered_gates + unused_gates


def add_to_gate(qubit, final_gate, links,
                ordered_gates, partial_orders):
    """
    Adds all gates that are not already in the ordered_gates
    list on the given qubit up to the final_gate, respecting
    any links.
    """
    links_to_cover = [[q2, g] for q2 in links[qubit]
                      for g in links[qubit][q2]]
    links_to_cover.sort(key=lambda x: x[1], reverse=True)

    # Add a fake link to prevent overrunning
    links_to_cover = [[None, final_gate+1]] + links_to_cover
    next_link = links_to_cover.pop()
    for gate in partial_orders[qubit]:
        if gate > final_gate:
            raise ValueError('Somehow Ive gotten past the end gate ' +
                             'on my qubit. This should never happen.')
        elif gate == final_gate:
            if gate not in ordered_gates:
                ordered_gates.append(gate)
            return
        elif gate == next_link[1]:
            add_to_gate(
                *next_link, links, ordered_gates, partial_orders)
            next_link = links_to_cover.pop()
        else:
            if gate not in ordered_gates:
                ordered_gates.append(gate)
    raise ValueError('The final gate given doesnt appear to be ' +
                     'a gate on this qubit. This should never happen.')
