import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from database import convert_fen_to_matrix
from keras.callbacks import TensorBoard




tensorboard_callback = TensorBoard(log_dir="./logs")

def load_dataset(path):
    dataset = tf.data.experimental.load(path)
    dataset = list(dataset.as_numpy_iterator())  # Convert dataset to a list of NumPy arrays

    # Split the dataset into training and test sets
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Access the features and label from the training dataset
    X_train = np.array([element[0] for element in dataset_train])
    y_train = np.array([element[1] for element in dataset_train])

    # Access the features and label from the test dataset
    X_test = np.array([element[0] for element in dataset_test])
    y_test = np.array([element[1] for element in dataset_test])

#     X_train = np.transpose(X_train)
#     X_test = np.transpose(X_test)
#    # Reshape the target labels if needed
#     y_train = y_train.reshape((1, y_train.shape[0]))
#     y_test = y_test.reshape((1, y_test.shape[0]))
    return X_train, y_train, X_test, y_test


def predict(fen):
    model = tf.keras.models.load_model('evalAI.model')
    matrix = convert_fen_to_matrix(fen)
    matrix = np.array(matrix)
    matrix = matrix.reshape(1, len(matrix))
    matrix = matrix / np.linalg.norm((matrix), axis=1, keepdims=True)

    # print('-------------------------------------')
    # print(matrix)
    # print(matrix.shape)
    # print('-------------------------------------')

    prediction = model.predict(matrix)
    return prediction





def main():
    load_path = '/home/karolito/DL/chess/dataset'
    X_train, y_train, X_test, y_test = load_dataset(load_path)

    print('-------------------------------------')
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(600, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(600, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(600, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(600, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(65, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[tensorboard_callback])

    answer = input("save? y/n")
    if answer ==("y"):
        model.save('evalAI.model')

    prediction = predict("8/4k1p1/4N2p/8/Bn6/2b1P1P1/p4P1P/7K w - - 4 33")
    print(prediction)




if __name__ == "__main__":
    tic = time.time()
    main()
    tac = time.time()
    print(f"Measured time: {tac - tic}")