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

num_iters = 0
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

class EvaluateProcess(multiprocessing.Process):
    def __init__(self, fen, position_count, num_iters):
        super(EvaluateProcess, self).__init__()
        self.fen = fen
        self.position_count = position_count
        self.num_iters = num_iters

    def run(self):
        # Your evaluation code here
        # ...

        directory = "/home/karolito/DL/chess/stockfish/stockfish_15.1_linux_x64_bmi2"
        file_name = "stockfish_15.1_x64_bmi2"
        stockfish_path = os.path.join(directory, file_name)
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(self.fen)
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            engine.quit()
            try:
                score = str(info["score"].white())
                int_score = int(score)
            except:
                int_score = 2000

            if "#" in score and "-" in score:
                int_score = -2000

            # print(int_score)
            # print(board)

            with self.num_iters.get_lock():
                self.num_iters.value += 1

            if self.num_iters.value % 10 == 0:  # Print every 10 iterations
                print(f"Number of iterations: {self.num_iters.value}")

        # Update the shared position count
        with self.position_count.get_lock():
            self.position_count.value += 1

        # Update the global variable
        with self.num_iters.get_lock():
            self.num_iters.value += 1

        # Print the number of iterations periodically
        if self.num_iters.value % 10 == 0:
            print(f"Number of iterations: {self.num_iters.value}")






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
    matrix.append(get_material_score(matrix[:-1]))
    return matrix


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
        return fens

    def write_to_txt(self, data, file_path):
        with open(file_path, 'w') as file:
            str_data = str(data)
            file.write(str_data)

def read_from_txt(file_path):
    with open(file_path, "r") as file:
        contents = file.read()
        my_list = eval(contents)
    return my_list

def get_piece_value(char):
    piece_map = {
        'k': -6, 'q': -5, 'r': -4, 'b': -3, 'n': -2, 'p': -1,
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1
        }
    return piece_map.get(char, 0)

def evaluate_position(fen):
    global num_iters


    directory = "/home/karolito/DL/chess/stockfish/stockfish_15.1_linux_x64_bmi2"
    file_name = "stockfish_15.1_x64_bmi2"
    stockfish_path = os.path.join(directory, file_name)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(time==0.02))
        engine.quit()
        try:
            score = str(info["score"].white())
            int_score = int(score)
        except:
            int_score = 2000

        if "#" in score and "-" in score:
            int_score = -2000


        return int_score

def stockfish_benchmark(time_1, time_2):
    fen_positions = [
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "3r2k1/1pp2pp1/2n4p/3p4/3P3P/1P1R1P2/P4P2/7K w - - 0 32",
        "r2qk2r/pppb1ppp/2n1pn2/3p4/2PP4/2N1P3/PP1N1PPP/R1BQ1RK1 b kq - 0 14",
        "r3r1k1/ppp2ppp/2p1b3/8/3Q4/8/PPP2PPP/R1B2RK1 b - - 0 14",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/3PP3/8/PPP2PPP/R1BQKBNR b KQkq - 1 5",
        "2r1r1k1/1q1n1p1p/p2P4/1p1p4/2pP1B2/1P5P/P2Q2P1/2R2RK1 w - - 0 26",
        "r1b1r1k1/1p1nqppp/p7/2pp4/3P4/2P1P1P1/PPQN1P1P/R1B1R1K1 w - - 0 21",
        "r2qkb1r/p1pnpppp/2n2n2/1p6/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7",
        "rnb1kb1r/ppp1pppp/8/3p4/3Pn3/5N2/PPP2PPP/R1BQKB1R w KQkq - 0 7",
        "r1bqkb1r/ppp1pppp/8/3p4/4n3/5N2/PPP2PPP/R1BQKB1R b KQkq - 0 7",
            "rnbq1rk1/pppp1ppp/4pn2/5b2/2B1P3/2N5/PPP1QPPP/2KR3R w - - 0 12",
    "rnbqkb1r/ppp2ppp/3p1n2/4p3/4P3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "3r2k1/pp3p2/2q2p1p/3np3/3N2P1/1P3P2/P5B1/1K2R2R w - - 0 24",
    "r2qk2r/ppp1bppp/2n2n2/3bp3/3N4/2P1B3/PPP1QPPP/R3K2R w KQkq - 0 11",
    "r2qk2r/ppp2ppp/2n5/4p3/4P3/2N1B3/PPP2PPP/R2Q1RK1 w kq - 0 11",
    "r2q1rk1/pp1nbpp1/3p2bp/3Pp3/3nP3/5N2/PPPQ1PPP/R1B1K2R w KQ - 1 15",
    "4r1k1/p4pp1/1p1n4/2q1P3/2p1N3/2Q3P1/PPP3PK/4R3 w - - 0 25",
    "r2qk2r/ppp1bpp1/2n2n1p/3bp1p1/3N3P/2P1P3/PPP2PB1/R1BQK1NR b KQkq - 0 6",
    "rnbqkb1r/pp2pppp/4Pn2/3p4/8/8/PPP2PPP/R1BQKBNR b KQkq - 0 5",
    "r2qkbnr/pp1b1ppp/2n1p3/3p2B1/3P4/8/PPP2PPP/RN1QK1NR w KQkq - 0 7"
    ]
    eval_difference = []

    directory = "/home/karolito/DL/chess/stockfish/stockfish_15.1_linux_x64_bmi2"
    file_name = "stockfish_15.1_x64_bmi2"
    stockfish_path = os.path.join(directory, file_name)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for fen in fen_positions:
            board = chess.Board(fen)
            eval_1 = engine.analyse(board, chess.engine.Limit(time=time_1))
            eval_2 = engine.analyse(board, chess.engine.Limit(time=time_2))
            variables = [eval_1, eval_2]

            for ix, var in enumerate(variables):
                if str(var["score"].white())[0] == "+":
                    variables[ix] = int(str(var["score"].white()).replace("+", ""))
                elif str(var["score"].white())[0] == "-":
                    variables[ix] = int(str(var["score"].white()).replace("-", ""))

            eval_1, eval_2 = variables
            eval_difference.append(abs(eval_1 - eval_2))

        engine.quit()

    return tf.reduce_mean(eval_difference)


def main():
    print("hey")
    database = Database()



    while True:
        directory = "/home/karolito/DL/chess/stockfish/stockfish_15.1_linux_x64_bmi2"
        file_name = "stockfish_15.1_x64_bmi2"
        stockfish_path = os.path.join(directory, file_name)
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board("2q1rk2/3nbppp/Q2p4/N3p3/4b3/2P1B1P1/2P2P1P/R4RK1 w - - 0 15")
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            engine.quit()
            try:
                score = str(info["score"].white())
                int_score = int(score)
            except:
                int_score = 2000

            if "#" in score and "-" in score:
                int_score = -2000
        print(int_score)

















    # fens = database.get_all_games("/home/karolito/DL/chess/Lichess_Elite_Database", 1_300_000)
    # fens = database.get_moves_from_pgn("/home/karolito/DL/chess/Lichess_Elite_Database/lichess_elite_2018-06.pgn", 600_000)
    # database.write_to_txt(fens, "data_test_new.txt")
    # print('hey')
    fens = read_from_txt("data_test_new.txt")

    # print(len(fens))
    # fens = read_from_txt("data.txt")
    matrices = [convert_fen_to_matrix(fen) for fen in fens]


    tic = time.time()
    print("Done")

    pool = multiprocessing.Pool(processes=6)

    results = pool.map(evaluate_position, fens)

    pool.close()
    pool.join()

    tac = time.time()
    print(f"Measured time: {tac - tic}")

    # Create the dataset with matrices and results
    dataset = tf.data.Dataset.from_tensor_slices((matrices, results))

    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, '18.12.New_data_600k')

    tf.data.experimental.save(dataset, save_path)


if __name__ == '__main__':

    main()



