import tensorflow as tf
import joblib
import time
import numpy as np
import pickle

import concurrent.futures


def expand_to_input(input_vector):
    chessboard = np.zeros((8, 8, 12), dtype=np.float32)

    piece_mapping = {
        'k': -6, 'q': -5, 'r': -4, 'b': -3, 'n': -2, 'p': -1,
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1
        }

    chessboard_vector = input_vector[:64]

    for i, piece_code in enumerate(chessboard_vector):
        if piece_code != 0:
            if piece_code < 0:
                channel_index = -piece_code + 5
            else:
                channel_index = piece_code - 1
            row = i // 8
            col = i % 8
            chessboard[row, col, channel_index] = 1.0

    # Extract whose turn it is (65th element)
    turn = input_vector[64]
    turn_matrix = np.ones((8, 8), dtype=np.float32) * turn

    # Extract the material score (last element)
    material_score = input_vector[-1]
    material_matrix = np.ones((8, 8), dtype=np.float32) * material_score
    input_tensor = np.dstack((chessboard, turn_matrix, material_matrix))

    return input_tensor

def expand_to_input_parallel(input_vector):
    return expand_to_input(input_vector)

def create_dataset(data, batch_size=512):
    def data_generator(data):
        for x in data:
            processed_data = expand_to_input(x)
            yield processed_data

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data),
        output_signature=tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32)
    )

    dataset = dataset.batch(batch_size)

    return dataset


def dataset_to_numpy(dataset):
    # Collect samples from the dataset
    X = []
    y = []
    for i, j in dataset:

        X.append(i)
        y.append(j)

    return np.array(X), np.array(y)


def convert_to_int32(element):
    # Assuming element is a tensor or a NumPy array
    return tf.cast(element, tf.int32)


def main():


    tf.config.run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only allocate memory as needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    print(gpus)
    

    #X_train = np.load('X_train_600k.npy')
    X_test = np.load('X_test_600k.npy')

    y_test = np.load("y_test_big.npy")
    print(len(y_test))
    print(len(X_test))
    print(X_test[100])
    print(y_test[100])
    return
    # tf_train = tf.data.Dataset.from_tensor_slices(X_train)
    tf_test = tf.data.Dataset.from_tensor_slices(X_test)
    # tf_train = tf_train.map(convert_to_float32).batch(512)
    tf_test = tf_test.map(convert_to_int32)


    # X_train = tf.data.Dataset.load("X_train_tf")

    X_test = tf.data.Dataset.load("X_test_tf")
    # for i in X_test.unbatch():
    #     print(i)
    # X_train = X_train.concatenate(tf_train)
    # del tf_train

    X_test = X_test.concatenate(tf_test)
    del tf_test
    print("damn")
    options = tf.data.Options()

    save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

    ds = np.load("y_train_big.npy")
    print(len(ds))
    # for i in ds:
    #     print(i)


if __name__ == "__main__":
    main()