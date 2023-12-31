import chess
from model import preprocess_input, custom_weighted_mse_loss
import tensorflow as tf
import time
import multiprocessing
import logging
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.utils import custom_object_scope
from convert_to_bitmap import expand_to_input
import numpy as np
from database import convert_fen_to_matrix, get_material_score



model = None
iters = 0
logging.basicConfig(level=logging.INFO, filename='output.log', filemode='w', format='%(message)s')


def get_legal_moves(fen):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    return moves, board


def convert_to_trt_model(input_model_path, output_model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(input_model_path)

    # Convert the Keras model to a TensorRT optimized model
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_model_path)
    converter.convert()
    converter.save(output_model_path)

def optimize_for_tensorrt(model):
    # Convert the Keras model to a TensorRT optimized model
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model, use_dynamic_shape=True)
    converter.convert()
    converter.save('trt_optimized_model')

def load_tensorrt_optimized_model(path):
    # Load the TensorRT optimized model
    trt_model = tf.saved_model.load(path)
    return trt_model

def predict_with_tensorrt(input_data, trt_model):
    engine_file = 'tensorrt_model/converted_model.plan'
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

def convert_eval_dict(eval):
    if type(eval) == dict:
        eval = eval["dense_3"]
        eval = eval.numpy()[0, 0]
    return eval

def search(board, depth, alpha, beta, color, model, max_depth=4, current_depth= 0, max_iters=20_000, max_time=20, time_start=0):
    global iters
    position = board.fen()

    if depth == 0 or current_depth > max_depth or max_iters < iters or (time.time() - time_start >= max_time):
        input = expand_to_input(convert_fen_to_matrix(position))
        return model(tf.reshape(input, (1, 8, 8, 14))), None

    moves_list, new_board = get_legal_moves(position)
    moves_list = order_move_list(moves_list, board)
    best_move = None
    if color:
        max_eval = float('-inf')
        for move in moves_list:
            attacking_piece, attackers, attacks = get_move_info(new_board, move)
            iters += 1
            board_2 = new_board.copy()
            board_2.push(move)
            if (is_capture(new_board, move) or is_a_blunder(attackers, attacking_piece, board_2) or is_a_threat(attacks, attacking_piece, board_2)) and (current_depth <= max_depth):
                new_board.push(move)
                evaluation, _ = search(new_board, depth, alpha, beta, False, model, current_depth=current_depth + 1, max_depth=max_depth, time_start=time_start, max_time=max_time, max_iters=max_iters)
                evaluation = convert_eval_dict(evaluation)

            else:
                new_board.push(move)
                evaluation, _ = search(new_board, depth - 1, alpha, beta, False, model, current_depth=current_depth + 1, max_depth=max_depth, time_start=time_start, max_time=max_time, max_iters=max_iters)
                evaluation = convert_eval_dict(evaluation)


            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move

            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break

            new_board.pop()

        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves_list:
            attacking_piece, attackers, attacks = get_move_info(new_board, move)
            iters += 1
            board_2 = new_board.copy()
            board_2.push(move)

            if (is_capture(new_board, move)  or is_a_blunder(attackers, attacking_piece, board_2) or is_a_threat(attacks, attacking_piece, board_2)) and (current_depth <= max_depth):
                new_board.push(move)
                evaluation, _ = search(new_board, depth, alpha, beta, True, model, current_depth=current_depth + 1, max_depth=max_depth , time_start=time_start, max_time=max_time, max_iters=max_iters)
                evaluation = convert_eval_dict(evaluation)

            else:
                new_board.push(move)
                evaluation, _ = search(new_board, depth - 1, alpha, beta, True, model, current_depth=current_depth + 1, max_depth= max_depth, time_start=time_start, max_time=max_time, max_iters=max_iters)
                evaluation = convert_eval_dict(evaluation)


            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move

            beta = min(beta, min_eval)
            if beta <= alpha:
                break

            new_board.pop()

        print(iters)
        return min_eval, best_move


def is_capture(board, move):
    return board.is_capture(move)

def is_a_threat(attacked_squares_set, piece, board):
    piece_value = {'P': 100, 'N': 310, 'B': 320, 'R': 500, 'Q': 900}  # Assign values to pieces based on their relative importance

    for square in attacked_squares_set:
        attacked_piece = board.piece_at(square)

        if attacked_piece is not None:
            if piece_value.get(attacked_piece.symbol(), 0) >= piece_value.get(piece.symbol(), 0):
                return True

    return False

def is_a_blunder(attackers, piece, board):
    piece_value = {'P': 100, 'N': 310, 'B': 320, 'R': 500, 'Q': 900}
    for square in attackers:
        attacker = board.piece_at(square)
        if attacker is not None and piece_value.get(attacker.symbol(), 0) < piece_value.get(piece.symbol(), 0):
            return True
    return False

def order_move_list(move_list, board):
    ordered_list = []
    scores = []
    score = 0
    for ix, move in enumerate(move_list):
        color = board.turn
        start = move.from_square
        attacking_piece = board.piece_at(start)
        destination = move.to_square
        attackers = board.attackers(not color, destination)
        attacks = board.attacks(destination)
        if board.is_capture(move):
            score += 100
        if board.gives_check(move):
            score += 300
        if is_a_threat(attacks, attacking_piece, board):
            score += 1000
        if is_a_blunder(attackers, attacking_piece, board):
            score -= 600
        scores.append((score, ix))
    scores.sort(key=lambda x: x[0], reverse=True)
    ordered_list = [move_list[i] for _, i in scores]

    return ordered_list


def load_opening_book(path):
    pass


def get_move_info(board, move):
        color = board.turn
        start = move.from_square
        attacking_piece = board.piece_at(start)
        destination = move.to_square
        attackers = board.attackers(not color, destination)
        attacks = board.attacks(destination)
        return attacking_piece, attackers, attacks


def speed_test(model):
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
        "r1bqkb1r/ppp1pppp/8/3p4/4n3/5N2/PPP2PPP/R1BQKB1R b KQkq - 0 7"
    ]
    times = []
    for fen in fen_positions:
        print("he")
        tic = time.time()
        model(tf.reshape(preprocess_input(fen), (1, 8, 8, 14)))
        tac = time.time()
        times.append(tac - tic)
    return times


def main():
    # tf.keras.utils.get_custom_objects()['custom_weighted_mse_loss'] = custom_weighted_mse_loss
    # model = tf.keras.models.load_model("model_model.model")
    # optimize_for_tensorrt("model_model.model")
    trt_model = load_tensorrt_optimized_model("trt_optimized_model")




    signature_keys = list(trt_model.signatures.keys())

    graph_func = trt_model.signatures[signature_keys[0]]



    position = "N1bk3r/pp2bppp/2n5/3p4/2n5/P7/1PPP1PPP/R1B1K1NR w KQ - 0 11"
    input = convert_fen_to_matrix(position)

    input = expand_to_input(input)
    input - tf.reshape(input, (8, 8, 14))
    print(input)
    return

    initial_pred = graph_func(tf.reshape(input, (1, 8, 8, 14)))
    # times = speed_test(graph_func)
    # times.pop(0)
    # print(times)
    # average_time = tf.reduce_mean(times)
    # print(average_time * 200_000 / 60)
    board = chess.Board(position)
    color = board.turn
    tic = time.time()
    print("hey")
    move = search(board=board, depth=2, alpha=float('-inf'), beta= float("inf"), color=color, model=graph_func, max_depth=4, current_depth=0, max_iters=30_000, max_time=300, time_start=tic)
    tac = time.time()
    print(f"Measured time: {tac - tic}")
    print(move)
    print(iters)
    print(initial_pred)


if __name__ == "__main__":

    main()
