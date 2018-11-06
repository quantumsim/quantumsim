import quantumsim.tp as tp
import unittest

test_data_dict = {1: set(),
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
test_data_list = [[11, 4, 12], [1, 13, 3, 14, 4, 15, 7, 16, 9],
                  [17, 3, 18, 5, 19], [20, 2, 21, 6, 22, 5, 23, 8, 24, 10],
                  [25, 6, 26]]




def test_greedy_list_toposort():
    result = tp.partial_greedy_toposort(test_data_list)

    print(result)
    indices = {s: i for i, s in enumerate(result)}


    assert set(result) == set(test_data_dict.keys())

    for s, s_set in test_data_dict.items():
        for s2 in s_set:
            assert indices[s2] < indices[s]


def test_regression():
    list = [[1], [0, 1, 2, 3, 5], [4, 5], [2, 3, 4]]
    target = [2]


    item_set = set([x for xs in list for x in xs])

    result = tp.partial_greedy_toposort(list, target)

    assert set(result) == item_set

    assert len(result) == len(item_set)


class TestSimpleToposort(unittest.TestCase):

    def test_add_to_gate_simple(self):
        qubit = 0
        final_gate = 5
        links = {0: {}}
        ordered_gates = []
        partial_orders = [[0, 1, 2, 3, 4, 5]]
        tp.add_to_gate(qubit, final_gate, links, ordered_gates, partial_orders)
        self.assertEqual(len(ordered_gates), 6)

    def test_add_to_gate_two_qubits_no_connection(self):
        qubit = 0
        final_gate = 3
        links = {0: {1: []}, 1: {0: []}}
        ordered_gates = []
        partial_orders = [[0, 2, 3], [1, 4, 5]]
        tp.add_to_gate(qubit, final_gate, links, ordered_gates, partial_orders)
        self.assertEqual(len(ordered_gates), 3)

    def test_add_to_gate_two_qubits_with_connection(self):
        qubit = 0
        final_gate = 3
        links = {0: {1: [2]}, 1: {0: [2]}}
        ordered_gates = []
        partial_orders = [[0, 2, 3], [1, 2, 4, 5]]
        tp.add_to_gate(qubit, final_gate, links, ordered_gates, partial_orders)
        print(ordered_gates)
        self.assertEqual(len(ordered_gates), 4)

    def test_simple_toposort_single(self):
        partial_orders = [[0]]
        targets = {0: 0}
        res = tp.simple_toposort(partial_orders, targets)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], 0)

    def test_simple_toposort_complex(self):
        partial_orders = [[0, 4, 5], [1, 3, 4, 6], [2, 3, 7]]
        targets = {0: 5, 2: 7}
        res = tp.simple_toposort(partial_orders, targets)
        print(res)
        self.assertEqual(len(res), 8)
        self.assertEqual(res[3], 7)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[0], 2)
        self.assertEqual(res[2], 3)
        self.assertEqual(res[4], 0)
        self.assertEqual(res[5], 4)
        self.assertEqual(res[6], 5)
        self.assertEqual(res[7], 6)
