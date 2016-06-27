import quantumsim.tp as tp

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
