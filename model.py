import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from database import convert_fen_to_matrix, get_material_score
from convert_to_bitmap import expand_to_input
from temporary import filter_second_to_last_element_equal_to_1

tensorboard_callback = TensorBoard(log_dir="./logs")
early_stopping_callback = EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(filepath="model_checkpoint.keras", monitor="val_loss", save_best_only=True)

def load_saved_dataset():
    X_train = np.load("X_train_expanded.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test_expanded.npy")
    y_test = np.load("y_test.npy")
    return X_train, y_train, X_test, y_test

def load_dataset(path):
    dataset = tf.data.experimental.load(path)
    dataset = list(dataset.as_numpy_iterator())
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=42)

    X_train = np.array([element[0] for element in dataset_train])
    y_train = np.array([element[1] for element in dataset_train])
    X_test = np.array([element[0] for element in dataset_test])
    y_test = np.array([element[1] for element in dataset_test])

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    return X_train, y_train, X_test, y_test

def predict(matrix, model):
    return model.predict(matrix)

def merge_datasets(database1, database2):
    return database1.concatenate(database2)

def preprocess_input(fen):
    matrix = convert_fen_to_matrix(fen)
    matrix = np.array(matrix).reshape(1, -1)
    return tf.convert_to_tensor(matrix, dtype=tf.float32)

def convert_array(path):
    dataset = tf.data.experimental.load(path)
    dataset = list(dataset.as_numpy_iterator())
    converted_dataset = []
    labels = []
    for matrix in dataset:
        material_score = get_material_score(matrix=matrix)
        element = np.append(matrix[0], material_score)
        converted_dataset.append(element)
        labels.append(matrix[1])
    return converted_dataset, labels

def custom_loss(y_true, y_pred, smoothing_factor=0.6):
    y_true_smooth = (1 - smoothing_factor) * y_true + smoothing_factor / y_true.shape[1]
    label_difference = tf.abs(y_true - y_pred)
    loss = tf.reduce_mean(label_difference**2 * y_true_smooth)
    return loss

def labeled_model(X_train, y_train, X_test, y_test, num_classes, save_path, save_model=True):
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[tensorboard_callback], batch_size=256)

    if save_model:
        model.save(save_path)

def custom_weighted_mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_abs = tf.abs(y_true)
    y_pred_abs = tf.abs(y_pred)
    squared_errors = tf.abs(tf.subtract(y_true, y_pred))
    mse_loss = tf.reduce_mean(squared_errors)

    condition = tf.math.less(tf.math.multiply(y_true, y_pred), 0)
    loss = tf.where(condition, mse_loss, tf.multiply(mse_loss, 7))

    condition_1 = tf.math.less(y_true_abs, 100)
    loss = tf.where(condition_1, mse_loss, tf.multiply(mse_loss, 4))

    condition_2 = tf.math.greater(y_true_abs, 800)
    condition_3 = tf.math.greater(y_pred_abs, 700)
    loss = tf.where([condition_2, condition_3], loss, tf.multiply(mse_loss, 0.3))

    return loss

def regular_mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_abs = tf.abs(y_true)
    squared_errors = tf.square(y_true - y_pred)
    mse_loss = tf.reduce_mean(squared_errors)
    condition = tf.math.less(y_true_abs, 300)
    loss = tf.where(condition, mse_loss, tf.multiply(mse_loss, 0.5))
    return loss

def reshape_feature_vector(feature_vector):
    return feature_vector.reshape(11, 2, 3)

def resnet_50_model(input_shape, fine_tune_at):
    base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def cast_to_int32(x, y):
    return tf.cast(x, tf.int32), y

def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - residual / total
    return r2

def load_train_ds(batch_size):
    ds = tf.data.Dataset.load("filtered_dataset").unbatch()
    ds = ds.filter(filter_second_to_last_element_equal_to_1)
    ds_white_1 = tf.data.Dataset.load("fixed_whites")
    ds_white_2 = tf.data.Dataset.load("fixed_whites_2")
    ds_white_1 = ds_white_1.map(cast_to_int32)
    ds_white_2 = ds_white_2.map(cast_to_int32)
    ds_merged = ds.concatenate(ds_white_1)
    ds_final = ds_merged.concatenate(ds_white_2).shuffle(50 * 256)
    return ds_final.batch(batch_size)

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.excitation_1 = tf.keras.layers.Dense(channels // ratio, activation="relu")
        self.excitation_2 = tf.keras.layers.Dense(channels, activation="sigmoid")
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        squeeze = self.squeeze(inputs)
        excitation_1 = self.excitation_1(squeeze)
        excitation_2 = self.excitation_2(excitation_1)
        reshape = self.reshape(excitation_2)
        output = self.multiply([inputs, reshape])
        return output

def convert_savedmodel_to_keras(saved_model_path, keras_model_path):
    loaded_model = tf.saved_model.load(saved_model_path)
    concrete_func = loaded_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_tensor = list(concrete_func.inputs.values())[0]
    input_shape = input_tensor.shape[1:]

    @tf.function
    def model_function(inputs):
        return concrete_func(inputs)

    inputs = tf.keras.Input(shape=input_shape)
    outputs = model_function(inputs)

    if isinstance(outputs, dict):
        outputs = list(outputs.values())[0]

    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    keras_model.compile(
        optimizer=optimizer,
        loss=custom_weighted_mse_loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.RootMeanSquaredError(),
        ]
    )

    keras_model.save(keras_model_path)

def main():

    tf.keras.utils.get_custom_objects()["custom_weighted_mse_loss"] = custom_weighted_mse_loss
    y_test = np.load("y_test_big.npy")
    X_test = tf.data.Dataset.load("saved_tf_test")
    train_ds = load_train_ds(512)

    X_test_unbatched = X_test.unbatch()
    y_test_unbatched = tf.data.Dataset.from_tensor_slices(y_test)
    test_ds = tf.data.Dataset.zip((X_test_unbatched, y_test_unbatched))
    test_ds_black = test_ds.filter(filter_second_to_last_element_equal_to_1)
    test_ds_white = tf.data.Dataset.load("new_test_ds")
    test_ds_white = test_ds_white.map(cast_to_int32)
    test_ds = test_ds_black.concatenate(test_ds_white).shuffle(20 * 256).batch(512)

    del X_test, X_test_unbatched, y_test, y_test_unbatched
    gc.collect()

    checkpoint_callback = ModelCheckpoint(
        filepath="weights_{epoch:02d}.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100000)]
            )
        except RuntimeError as e:
            print(e)

    model = create_model()
    # model.load_weights("best_weights.weights.h5")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(
        optimizer=optimizer,
        loss=custom_weighted_mse_loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.RootMeanSquaredError(),
            r_squared,
        ],
    )
    model.fit(train_ds, validation_data=test_ds, epochs=100, callbacks=[checkpoint_callback])
    model.save("model_keras.h5")

    pred = model.predict(
        tf.reshape(
            expand_to_input(
                convert_fen_to_matrix(
                    "rnbqkbnr/p1pppppp/8/1p4P1/8/8/PPPPPP1P/RNBQKBNR b KQkq - 0 2"
                )
            ),
            (1, 8, 8, 14),
        )
    )
    print(pred)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(8, 8, 14)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5_000, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(5000, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(2000, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(700, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(600, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(500, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.built = True
    for layer in model.layers:
        layer.trainable = True
        print(layer.name, layer.trainable)
    model.summary()

    return model

if __name__ == "__main__":
    main()
