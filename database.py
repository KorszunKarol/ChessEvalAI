import os
import chess.pgn
import chess.engine
from io import StringIO
import sys
import time
import numpy as np
import tensorflow as tf
import re
import multiprocessing


def convert_fen_to_matrix(fen):
    matrix = []
    iterator = iter(fen)
    for char in iterator:
        if char == '/':
            continue
        if char == ' ':
            next_char = next(iterator, None)
            if next_char == "w":
                matrix.append(1)
            else:
                matrix.append(0)
            break

        try:
            num = int(char)
            for i in range(num):
                matrix.append(0)
        except ValueError:
            piece = get_piece_value(char)
            matrix.append(piece)
    return matrix


class Game:
    def __init__(self, moves) -> None:
        self.moves = moves


class Database:
    def __init__(self):
        self.positions = []
        self.evals = []

    def insert_postion(self, position):
        self.positions.append(position)

    def insert_eval(self, eval):
        self.evals.append(eval)

    def get_evals(self):
        return self.evals

    def get_positions(self):
        return self.positions

    def get_moves_from_pgn(self, file_path, size):
        fens = []
        with open(file_path) as pgn_file:
            for game in pgn_file:
                game = chess.pgn.read_game(pgn_file)

                board = game.board()

                for move in game.mainline_moves():
                    board.push(move)
                    position = board.fen()

                    # eval = evaluate_position(position)
                    # print(eval)
                    # print("\n")

                    matrix_pos = convert_fen_to_matrix(position)

                    if len(self.get_positions()) <= size:
                        # print(len(self.get_positions()))
                        self.insert_postion(matrix_pos)
                        # self.insert_eval(eval)
                        fens.append(position)
                    else:
                        return fens


    def get_all_games(self, folder_path, size):
        fens = []
        for file_name in os.listdir(folder_path):
            if size <= len(self.get_positions()):
                return fens
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_path:
                fen = self.get_moves_from_pgn(file_path, size)
                if fen not in fens:
                    fens.extend(fen)


    def write_to_txt(self, data, file_path):
        with open(file_path, 'w') as file:
            str_data = str(data)
            file.write(str_data)


def get_piece_value(char):
    piece_map = {
        'k': -6, 'q': -5, 'r': -4, 'b': -3, 'n': -2, 'p': -1,
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1
        }
    return piece_map.get(char, 0)

def evaluate_position(fen):
    with chess.engine.SimpleEngine.popen_uci(r"/home/karolito/DL/chess/stockfish/stockfish_15.1_linux_x64_bmi2/stockfish_15.1_x64_bmi2") as engine:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=17))
        engine.quit()
        try:
            score = str(info["score"].white())
            int_score = int(score)
        except:
            int_score = 2000

        print(int_score)
        print(board)
        if "#" in score and "-" in score:
            int_score = -2000
        print("a")
        return int_score

def main():
    db = Database()
    fens = db.get_all_games("/home/karolito/DL/chess/Lichess_Elite_Database", 100)

    games = db.get_positions()
    
    pool = multiprocessing.Pool(processes=16)

    # Perform analysis using multiprocessing
    results = pool.map(evaluate_position, fens)

    # Close the multiprocessing.Pool
    pool.close()
    pool.join()

    dataset = tf.data.Dataset.from_tensor_slices((games, results))

    # Save the dataset to a TensorFlow database
    # save_path = '/home/karolito/DL/chess/dataset'
    # tf.data.experimental.save(dataset, save_path)

    # load_path = '/home/karolito/DL/chess/dataset'
    # dataset = tf.data.experimental.load(load_path)

    # # Iterate over the dataset and print some samples
    # for game, result in dataset:
    #     print("Game:", game)
    #     print("Result:", result)
    #     print()




if __name__ == '__main__':
    tic = time.time()

    main()

    tac = time.time()
    print(f"Measured time: {tac - tic}")