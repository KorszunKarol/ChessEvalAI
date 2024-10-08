import os
import random
import chess
import chess.pgn
import io
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Generator, Tuple
from feature_engineering import *


@dataclass
class PGNProcessor:
    pgn_file_path: str
    chunk_size: int = 1_000_000_000
    progress_file: str = field(init=False)
    file_size: int = field(init=False)
    min_move_number: int = 8 

    def __post_init__(self):
        self.progress_file = os.path.join(os.path.dirname(self.pgn_file_path), 'progress.txt')
        self.file_size = os.path.getsize(self.pgn_file_path)

    def get_random_start_position(self) -> int:
        return random.randint(0, max(0, self.file_size - self.chunk_size))

    def find_next_game_start(self, file: io.TextIOWrapper, start_pos: int) -> Optional[int]:
        file.seek(start_pos)
        while True:
            line = file.readline()
            if not line:
                return None
            if line.strip().startswith('[Event '):
                return file.tell() - len(line)

    def process_games(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        start_pos = self.load_progress()

        with open(self.pgn_file_path, "r") as pgn_file:
            while start_pos < self.file_size:
                logging.info(f"Processing chunk starting at position {start_pos}")

                next_game_start = self.find_next_game_start(pgn_file, start_pos)
                if next_game_start is None:
                    break

                pgn_file.seek(next_game_start)
                game = chess.pgn.read_game(pgn_file)

                if game is not None:
                    board = game.board()
                    move_count = 0
                    for move in game.mainline_moves():
                        board.push(move)
                        move_count += 1

                        if move_count >= 10 or random.random() < 0.05:
                            fen = board.fen()
                            matrix = self.convert_fen_to_matrix(fen)
                            yield fen, matrix

                start_pos = pgn_file.tell()
                self.save_progress(start_pos)

    def convert_fen_to_matrix(self, fen: str) -> np.ndarray:
        matrix = np.zeros((8, 8, 60), dtype=np.float32)  # Updated to 60 channels
        board = chess.Board(fen)
        piece_to_index = {
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
            'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
        }

        # Set piece positions (channels 0-11)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                row, col = divmod(i, 8)
                matrix[row, col, piece_to_index[piece.symbol()]] = 1

        # Set the turn (channel 12)
        matrix[:, :, 12] = 1 if board.turn == chess.WHITE else 0

        # Set castling rights (channels 13-16)
        matrix[:, :, 13] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        matrix[:, :, 14] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        matrix[:, :, 15] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        matrix[:, :, 16] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

        # Calculate and set material score (channel 17)
        material_score = self.calculate_material_score(board)
        matrix[:, :, 17] = material_score

        # En Passant Square (channel 18)
        ep_square = board.ep_square
        if ep_square is not None:
            row, col = divmod(ep_square, 8)
            matrix[row, col, 18] = 1

        # Half-move clock (channel 19)
        matrix[:, :, 19] = board.halfmove_clock / 100  # Normalize to 0-1 range

        # Full move number (channel 20)
        matrix[:, :, 20] = board.fullmove_number / 200  # Normalize assuming max 200 moves

        # Piece mobility (channel 21)
        matrix[:, :, 21] = get_piece_mobility(board)

        # Pawn structure (channels 22, 23, 24)
        doubled, isolated, passed = analyze_pawn_structure(board)
        matrix[:, :, 22] = doubled
        matrix[:, :, 23] = isolated
        matrix[:, :, 24] = passed

        # Center control (channel 26)
        matrix[:, :, 26] = center_control(board)

        # Piece-square tables (channel 27)
        matrix[:, :, 27] = piece_square_tables(board)

        # Defended and Vulnerable (channels 29-30)
        defended, vulnerable = defended_and_vulnerable(board)
        matrix[:, :, 29] = defended[:, :, 0]  # White
        matrix[:, :, 30] = defended[:, :, 1]  # Black
        matrix[:, :, 31] = vulnerable[:, :, 0]  # White
        matrix[:, :, 32] = vulnerable[:, :, 1]  # Black

        # Piece Coordination (channel 33)
        matrix[:, :, 33] = piece_coordination(board)

        # Game Phase (channel 59)
        matrix[:, :, 59] = game_phase(board)

        return matrix

    @staticmethod
    def calculate_material_score(board: chess.Board) -> float:
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        score = 0
        for piece_type in piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        return score / 39  # Normalize by maximum possible material difference

    def load_progress(self) -> int:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return int(f.read().strip())
        return self.get_random_start_position()

    def save_progress(self, position: int):
        with open(self.progress_file, 'w') as f:
            f.write(str(position))

    def reset_progress(self):
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        logging.info("Progress reset.")
