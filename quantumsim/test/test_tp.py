from quantumsim.tp import greedy_toposort


def test_greedy_toposort():
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
    solution = [1, 17, 13, 3, 11, 14, 4, 15, 7, 16,
            9, 20, 2, 21, 25, 6, 18, 22, 5, 23, 8, 24, 10, 26, 19, 12] 
    assert solution == greedy_toposort(test_data, {9, 10})
