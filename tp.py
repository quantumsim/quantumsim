import pytools

test_data = {1: set(),
             2: {20},
             3: {17,
                 13},
             4: {14,
                 11},
             5: {22,
                 18},
             6: {21,
                 25},
             7: {15},
             8: {23},
             9: {16},
             10: {24},
             11: set(),
             12: {4},
             13: {1},
             14: {3},
             15: {4},
             16: {7},
             17: set(),
             18: {3},
             19: {5},
             20: set(),
             21: {2},
             22: {6},
             23: {5},
             24: {8},
             25: set(),
             26: {6}}


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
    # the implementation is probably rather slow, so don't try this on big graphs


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


print(greedy_toposort(test_data, {9, 10}))
