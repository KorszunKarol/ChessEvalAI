import chess
import numpy as np

def get_piece_mobility(board):
    """
    Calculates the mobility of each piece on the board for both White and Black sides.
    Positive values indicate White pieces' mobility, and negative values indicate Black pieces' mobility.

    Args:
        board (chess.Board): The current state of the chess board.

    Returns:
        np.ndarray: An 8x8 matrix representing the mobility of each piece on the board.
    """
    mobility = np.zeros((8, 8), dtype=np.float32)

    # Define maximum moves per piece type for normalization
    max_moves_per_piece = {
        chess.PAWN: 4,
        chess.KNIGHT: 8,
        chess.BISHOP: 13,
        chess.ROOK: 14,
        chess.QUEEN: 27,
        chess.KING: 8
    }

    def calculate_mobility(board_copy, multiplier):
        """
        Calculates mobility for the current player in board_copy and updates the mobility matrix.

        Args:
            board_copy (chess.Board): A copy of the chess board.
            multiplier (int): 1 for White, -1 for Black.
        """
        legal_moves = list(board_copy.legal_moves)
        for move in legal_moves:
            from_square = move.from_square
            piece = board_copy.piece_at(from_square)
            if piece:
                # Correct row mapping: invert the row index
                original_row, col = divmod(from_square, 8)
                row = 7 - original_row  # Invert row index
                piece_type = piece.piece_type
                max_moves = max_moves_per_piece.get(piece_type, 27)
                mobility[row, col] += multiplier / max_moves

    # Calculate mobility for White
    calculate_mobility(board, 1)

    # Create a copy of the board and switch turn to calculate mobility for Black
    board_copy = board.copy()
    board_copy.turn = not board.turn
    calculate_mobility(board_copy, -1)

    mobility = mobility * -1

    return mobility

def analyze_pawn_structure(board):
    doubled = np.zeros((8, 8), dtype=np.float32)
    isolated = np.zeros((8, 8), dtype=np.float32)
    passed = np.zeros((8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            file, rank = chess.square_file(square), chess.square_rank(square)
            color = piece.color

            # Doubled pawns
            if any(board.piece_at(chess.square(file, r)) == piece for r in range(rank + 1, 8)):
                doubled[rank, file] = 1

            # Isolated pawns
            if not any(board.piece_at(chess.square(f, r)) == piece
                       for f in [file - 1, file + 1] if 0 <= f < 8
                       for r in range(8)):
                isolated[rank, file] = 1

            # Passed pawns
            if not any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, not color)
                       for f in [file - 1, file, file + 1] if 0 <= f < 8
                       for r in range(rank + (1 if color else -1), 8 if color else -1, 1 if color else -1)):
                passed[rank, file] = 1

    return doubled, isolated, passed

def defented_and_vulnerable(board):
    defended = np.zeros((8, 8, 2), dtype=np.float32)  # [0] White, [1] Black
    vulnerable = np.zeros((8, 8, 2), dtype=np.float32)

    for color in [chess.WHITE, chess.BLACK]:
        opponent_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                is_defended = any(board.attackers(color, attacker_square)
                                  for attacker_square in board.attackers(color, square))
                is_vulnerable = not is_defended
                row, col = divmod(square, 8)
                defended[row, col, int(color)] = 1 if is_defended else 0
                vulnerable[row, col, int(color)] = 1 if is_vulnerable else 0
    return defended, vulnerable

def piece_coordination(board):
    coordination = np.zeros((8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(piece.color, square)
            coordination[chess.square_rank(square), chess.square_file(square)] = len(attackers) / 4  # Normalize
    return coordination

def piece_square_tables(board):
    pst = np.zeros((8, 8), dtype=np.float32)
    piece_values = {
        chess.PAWN: [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ],
        chess.KNIGHT: [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50,
        ],
        chess.BISHOP: [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20,
        ],
        chess.ROOK: [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ],
        chess.QUEEN: [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ],
        chess.KING: [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_type = piece.piece_type
            color = piece.color
            if color == chess.BLACK:
                row = 7 - row
            if piece_type in piece_values:
                pst[row, col] = piece_values[piece_type][square] / 100  # Normalize
            else:
                pst[row, col] = 0  # Default value for unknown piece types

    return pst

def advanced_piece_square_tables(board):
    return piece_square_tables(board), piece_square_tables(board), piece_square_tables(board)

def is_outpost(board, square, color):
    opponent_pawns = chess.PAWN
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    opponent_color = not color
    pawn_attack_squares = [
        chess.square(file - 1, rank + 1) if color == chess.WHITE else chess.square(file - 1, rank - 1),
        chess.square(file + 1, rank + 1) if color == chess.WHITE else chess.square(file + 1, rank - 1)
    ]
    for pawn_square in pawn_attack_squares:
        if 0 <= pawn_square < 64 and board.piece_at(pawn_square) and board.piece_at(pawn_square).piece_type == opponent_pawns and board.piece_at(pawn_square).color == opponent_color:
            return False
    return True

def center_control(board):
    control = np.zeros((8, 8), dtype=np.float32)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    for square in center_squares:
        row, col = divmod(square, 8)
        control[row, col] = len(board.attackers(chess.WHITE, square)) - len(board.attackers(chess.BLACK, square))
    return control / 4  # Normalize

def defended_and_vulnerable(board):
    defended = np.zeros((8, 8, 2), dtype=np.float32)  # [0] White, [1] Black
    vulnerable = np.zeros((8, 8, 2), dtype=np.float32)

    for color in [chess.WHITE, chess.BLACK]:
        opponent_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                is_defended = any(board.attackers(color, attacker_square)
                                  for attacker_square in board.attackers(color, square))
                is_vulnerable = not is_defended
                row, col = divmod(square, 8)
                defended[row, col, int(color)] = 1 if is_defended else 0
                vulnerable[row, col, int(color)] = 1 if is_vulnerable else 0
    return defended, vulnerable

def game_phase(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    total_material = sum(len(board.pieces(piece_type, color)) * value
                         for piece_type, value in piece_values.items()
                         for color in [chess.WHITE, chess.BLACK])

    # Assuming max material is 78 (16 pawns, 4 knights, 4 bishops, 4 rooks, 2 queens)
    max_material = 78

    # Normalize the phase between 0 (opening) and 1 (endgame)
    phase = 1 - (total_material / max_material)
    phase = np.full((8, 8), phase, dtype=np.float32)

    return phase
