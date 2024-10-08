import chess
import numpy as np
from feature_engineering import get_piece_mobility
import tensorflow
import os


def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cpu = tensorflow.config.list_physical_devices('CPU')
    print(f"CPU devices: {cpu}")
    if cpu:
        tensorflow.config.set_visible_devices(cpu, 'CPU')
        print("Running on CPU")
    else:
        print("No CPU devices found")

    board = chess.Board(fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

    # Print the board for reference
    print("Initial Board:")
    print(board)

    # Call the get_piece_mobility_separate function
    mobility = get_piece_mobility(board)
    print(mobility)

if __name__ == "__main__":
    test()