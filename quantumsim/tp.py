# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt
import pytools


def greedy_toposort(data, targets):
    """Perform a topological sort. However, certain nodes, called `targets' are prioritized,
    so that they occur in the total order as early as possible.
    A greedy scheme is employed to achieve this.

    data: a dictionary of the form {node: set(dependency_node1, dependency_node2, ...), ...}
    target: a subset of data.keys(), defining the target nodes

    Returns a list containing each element of data.keys(), in such an order that
    all nodes occurs after all their dependency nodes listed in data, and
    target nodes appear as soon as possible.
    """

    # NOTES: So far this only works on dependency graphs without loops, and
    # the implementation is probably rather slow, so don't try this on big
    # graphs

    # first we find all terminal nodes in the graph.
    all_nodes = set(data.keys())
    non_terminal_nodes = set(pytools.flatten(data.values()))
    terminal_nodes = all_nodes - non_terminal_nodes
    other_terminal_nodes = terminal_nodes - targets

    assert targets.issubset(all_nodes)

    # we construct the trees
    # for all the targets
    target_trees = {}
    for target in targets:
        target_trees[target] = [{target}]
        all_nodes_in_tree = {target}
        while target_trees[target][-1]:
            next_level = set()
            for node in target_trees[target][-1]:
                next_level |= data[node]
            next_level.difference_update(all_nodes_in_tree)
            all_nodes_in_tree |= next_level
            target_trees[target].append(next_level)
        del target_trees[target][-1]

    # and the trees for all the other terminal nodes
    other_node_trees = {}
    for node in other_terminal_nodes:
        other_node_trees[node] = [{node}]
        all_nodes_in_tree = {node}
        while other_node_trees[node][-1]:
            next_level = set()
            for node2 in other_node_trees[node][-1]:
                next_level |= data[node2]
            next_level.difference_update(all_nodes_in_tree)
            all_nodes_in_tree |= next_level
            other_node_trees[node].append(next_level)
        del other_node_trees[node][-1]

    result = []

    while target_trees:
        # calculate the size of each target tree
        target_tree_size = {}
        for target in target_trees:
            tree = target_trees[target]
            target_tree_size[target] = sum(len(x) for x in tree)

        # find the smallest tree
        target, size = min(target_tree_size.items(), key=(lambda x: x[1]))
        treenodes = list(pytools.flatten(reversed(target_trees[target])))

        # append to result and remove from targets
        result.extend(treenodes)
        del target_trees[target]

        # remove added nodes from all other trees
        treenodes_set = set(treenodes)
        for target in target_trees:
            for x in target_trees[target]:
                x.difference_update(treenodes_set)
        for node in other_node_trees:
            for x in other_node_trees[node]:
                x.difference_update(treenodes_set)

    while other_node_trees:
        node, tree = other_node_trees.popitem()
        treenodes = list(pytools.flatten(reversed(tree)))

        # append to result and remove from targets
        result.extend(treenodes)

        # remove added nodes from all other trees
        treenodes_set = set(treenodes)
        for node in other_node_trees:
            for x in other_node_trees[node]:
                x.difference_update(treenodes_set)

    return result


def partial_greedy_toposort(partial_orders):
    """Given a list of partial orders [p1, p2, ...] of hashable items pi = [a_i0, a_i1, ...],
    representing the constraints

        a_i0 < a_i1 < ...

    construct a total ordering P: i0 < i1 < i2 < ...
    satisfying the following the following minimization problem:

    For each i in P, construct the set of P_i = {pi| a_i0 <= i <= ai_n-1 }.
    Then minimize

    max_i | P_i |

    (morally: try to completely embed one list before going for the next)

    This is done by a greedy algorithm.
    """

    order_dicts = []
    for n, p in enumerate(partial_orders):
        order_dict = {i: j for i, j in zip(p[1:], p)}
        order_dicts.append(order_dict)

    trees = []
    for n, p in enumerate(partial_orders):
        lists_used = set()
        tree = []
        to_do = [(n, p[-1])]
        while to_do:
            n, x = to_do.pop()
            tree.append((n, x))
            for n2, o2 in enumerate(order_dicts):
                x2 = o2.get(x)
                if x2 is not None:
                    to_do.append((n2, x2))

        lists_used = {n for n, x in tree}
        trees.append((tree, len(lists_used)))

    result = []
    while trees != []:
        trees.sort(key=lambda xy: xy[1], reverse=True)
        print(smallest[1])
        smallest = smallest[0]
        smallest.reverse()
        smallest = [x for n, x in smallest]

        new_trees = []
        for l, i in trees:
            l2 = [(n, x) for n, x in l if x not in smallest]
            i = len({n for n, x in l2})
            new_trees.append((l2, i))

        trees = new_trees

        result.extend(smallest)

    return result
