import chess
from model import preprocess_input, custom_weighted_mse_loss, cast_to_int32
import tensorflow as tf
import time
import multiprocessing
import logging
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.utils import custom_object_scope
from convert_to_bitmap import expand_to_input
import numpy as np
from database import convert_fen_to_matrix, get_material_score
from temporary import filter_second_to_last_element_equal_to_1
from tensorflow.keras import mixed_precision
import os

model = None
iters = 0
initial_depth = None
logging.basicConfig(
    level=logging.INFO, filename="output.log", filemode="w", format="%(message)s"
)


def get_legal_moves(fen):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    return moves, board


def calibration_input_fn():
    y_test = np.load("y_test_big.npy")

    X_test = tf.data.Dataset.load("saved_tf_test")
    X_test_unbatched = X_test.unbatch()
    y_test_unbatched = tf.data.Dataset.from_tensor_slices(y_test)
    test_ds = tf.data.Dataset.zip((X_test_unbatched, y_test_unbatched))
    test_ds_black = test_ds.filter(filter_second_to_last_element_equal_to_1)
    test_ds_white = tf.data.Dataset.load("new_test_ds")
    test_ds_white = test_ds_white.map(lambda x, y: (tf.cast(x, tf.int32), y))

    test_ds = (
        test_ds_black.concatenate(test_ds_white)
        .shuffle(5 * 256)
        .map(lambda x, y: (tf.cast(x, tf.float32), y))
    )
    test_ds = test_ds.map(lambda x, y: (tf.reshape(x, (1, 8, 8, 14))))

    batch_size = 1
    x = test_ds[0:batch_size, :]
    yield [x]


def calibration_input_fn():
    X_test = tf.data.Dataset.load("saved_tf_test").take(30)
    for x in X_test:
        yield [x]


def optimize_for_tensorrt(model, name):
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model,
        precision_mode=trt.TrtPrecisionMode.FP16,
        use_dynamic_shape=False,
        max_workspace_size_bytes=10000 * 1024 * 1024 * 1024,
    )
    converter.convert()

    converter.save(name)


def load_tensorrt_optimized_model(path):

    assert os.path.exists(path)
    print("Reading engine from file {}".format(path))
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

    return trt_model


def predict_with_tensorrt(input_data, trt_model):
    engine_file = "tensorrt_model/converted_model.plan"
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())


def convert_eval_dict(eval):
    if type(eval) == dict:
        eval = eval["dense_8"]
        eval = eval.numpy()[0, 0]
    return eval


def search(
    board,
    depth,
    alpha,
    beta,
    color,
    model,
    max_depth=3,
    current_depth=0,
    max_iters=20_000,
    max_time=20,
    time_start=0,
    move_order_list=[],
):
    global iters
    global initial_depth
    best_moves = []
    position = board.fen()
    if (
        depth == 0
        or current_depth > max_depth
        or max_iters < iters
        or (time.time() - time_start >= max_time)
    ):
        input = expand_to_input(convert_fen_to_matrix(position))
        return model(tf.reshape(input, (1, 8, 8, 14))), None, None

    best_move = None
    if color:
        value = float("-inf")
        if len(move_order_list) == 0:
            moves_list, new_board = get_legal_moves(position)
            moves_list = order_move_list(moves_list, board)
        elif current_depth == 0:
            moves_list = move_order_list

        for move in moves_list:
            attacking_piece, attackers, attacks = get_move_info(new_board, move)
            iters += 1
            board_2 = new_board.copy()
            board_2.push(move)
            if (
                (
                    (
                        is_capture(new_board, move)
                        or is_a_threat(attacks, attacking_piece, board_2)
                    )
                    and not is_a_blunder(attackers, attacking_piece, board_2)
                )
                and (current_depth < max_depth and current_depth > 0)
                and max_depth != initial_depth
            ):
                new_board.push(move)
                evaluation, _, _ = search(
                    new_board,
                    depth,
                    alpha,
                    beta,
                    False,
                    model,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    time_start=time_start,
                    max_time=max_time,
                    max_iters=max_iters,
                )
                evaluation = convert_eval_dict(evaluation)

            else:
                new_board.push(move)
                evaluation, _, _ = search(
                    new_board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    model,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    time_start=time_start,
                    max_time=max_time,
                    max_iters=max_iters,
                )
                evaluation = convert_eval_dict(evaluation)

            new_board.pop()

            if evaluation > value:
                value = evaluation
                best_move = move
            best_moves.append((move, evaluation))

            if value >= beta:
                break
            alpha = max(alpha, value)
    else:
        value = float("inf")
        if len(move_order_list) == 0:
            moves_list, new_board = get_legal_moves(position)
            moves_list = order_move_list(moves_list, board)
        elif current_depth == 0:
            moves_list = move_order_list

        for move in moves_list:
            attacking_piece, attackers, attacks = get_move_info(new_board, move)
            iters += 1
            board_2 = new_board.copy()
            board_2.push(move)

            if (
                (
                    (
                        is_capture(new_board, move)
                        or is_a_threat(attacks, attacking_piece, board_2)
                    )
                    and not is_a_blunder(attackers, attacking_piece, board_2)
                )
                and (current_depth < max_depth and current_depth > 0)
                and max_depth != initial_depth
            ):
                new_board.push(move)
                evaluation, _, _ = search(
                    new_board,
                    depth,
                    alpha,
                    beta,
                    True,
                    model,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    time_start=time_start,
                    max_time=max_time,
                    max_iters=max_iters,
                )
                evaluation = convert_eval_dict(evaluation)

            else:
                new_board.push(move)
                evaluation, _, _ = search(
                    new_board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    model,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    time_start=time_start,
                    max_time=max_time,
                    max_iters=max_iters,
                )
                evaluation = convert_eval_dict(evaluation)

            new_board.pop()

            if evaluation < value:
                value = evaluation
                best_move = move
            best_moves.append((move, evaluation))

            if value <= alpha:
                break
            beta = min(beta, value)
    print(iters)
    return value, best_move, best_moves


def is_capture(board, move):
    return board.is_capture(move)


def is_a_threat(attacked_squares_set, piece, board):
    piece_value = {"P": 100, "N": 310, "B": 320, "R": 500, "Q": 900}

    for square in attacked_squares_set:
        attacked_piece = board.piece_at(square)

        if attacked_piece is not None:
            if piece_value.get(attacked_piece.symbol(), 0) > piece_value.get(
                piece.symbol(), 0
            ):
                return True

    return False


def is_a_blunder(attackers, piece, board):
    piece_value = {"P": 100, "N": 310, "B": 320, "R": 500, "Q": 900}
    for square in attackers:
        attacker = board.piece_at(square)
        if attacker is not None and piece_value.get(
            attacker.symbol(), 0
        ) < piece_value.get(piece.symbol(), 0):
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
            score += 100
        if is_a_threat(attacks, attacking_piece, board):
            score += 100
        if is_a_blunder(attackers, attacking_piece, board):
            score -= 500
        scores.append((score, ix))
    scores.sort(key=lambda x: x[0], reverse=True)
    ordered_list = [move_list[i] for _, i in scores]

    return ordered_list


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
        "r1bqkb1r/ppp1pppp/8/3p4/4n3/5N2/PPP2PPP/R1BQKB1R b KQkq - 0 7",
    ]
    times = []
    for fen in fen_positions:
        tic = time.time()
        input = expand_to_input(convert_fen_to_matrix(fen))
        model(tf.reshape(input, (1, 8, 8, 14)))
        tac = time.time()
        times.append(tac - tic)
    return times


def iterative_deepening_search(
    board, model, max_depth=3, max_time=200, max_iters=200_000
):
    global iters
    global initial_depth

    color = board.turn
    best_move = None
    best_evaluation = float("-inf")
    best_move_list = []

    time_start = time.time()
    for depth in range(1, max_depth + 1):
        print("hey")
        evaluation, move, best_move_list = search(
            board,
            depth=depth,
            alpha=float("-inf"),
            beta=float("inf"),
            color=color,
            model=model,
            max_depth=depth,
            max_iters=100000,
            max_time=max_time,
            move_order_list=best_move_list,
            current_depth=0,
            time_start=time_start,
        )
        print("hey")
        print(evaluation)
        print(move)
        print(best_move_list)
        eval_prev = evaluation
        move_prev = move
        best_move_list = sorted(
            best_move_list, key=lambda x: x[1], reverse=True if color else False
        )
        best_move_list = list(map(lambda x: x[0], best_move_list))
        print(best_move_list)

        if time.time() - time_start >= max_time:
            break
    return eval_prev, move_prev, best_move_list


def main():

    trt_model = tf.saved_model.load("model_trt")
    signature_keys = list(trt_model.signatures.keys())
    graph_func = trt_model.signatures[signature_keys[0]]

    position = "r2qk2r/pp2bpp1/2P1pn2/1B4p1/8/3P1Q1P/PP3PP1/RN2R1K1 b kq - 0 13"
    input = convert_fen_to_matrix(position)

    input = expand_to_input(input)
    input - tf.reshape(input, (1, 8, 8, 14))
    print(input)
    initital_pred = graph_func(tf.reshape(input, (1, 8, 8, 14)))

    board = chess.Board(position)
    color = board.turn
    tic = time.time()
    evaluation, best_move, move_list = search(
        board=board,
        depth=3,
        alpha=float("-inf"),
        beta=float("inf"),
        color=color,
        model=graph_func,
        max_depth=3,
        current_depth=0,
        max_iters=300_000,
        max_time=300000,
        time_start=tic,
    )
    print(sorted(move_list, key=lambda x: x[1], reverse=True if color else False))

    return


if __name__ == "__main__":

    main()
