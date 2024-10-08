import numpy as np

def get_piece_value(char):
    piece_map = {
        'k': -6, 'q': -5, 'r': -4, 'b': -3, 'n': -2, 'p': -1,
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1
    }
    return piece_map.get(char, 0)

def get_material_score(matrix):
    sum_white = 0
    sum_black = 0

    vals = {
        -5: 9, -4: 5, -3: 3, -2: 3, -1: 1,
        5: 9, 4: 5, 3: 3, 2: 3, 1: 1
    }
    for field in matrix:
        if field < 0:
            if np.abs(field) != 6:
                sum_black += vals[field]
        elif field > 0:
            if np.abs(field) != 6:
                sum_white += vals[field]

    return sum_white - sum_black