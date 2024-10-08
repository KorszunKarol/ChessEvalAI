import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import random
import gc


def load_features_and_labels(dataset):


    # Split the dataset into training and test sets
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.05, random_state=42)

    # Access the features and label from the training dataset
    X_train = np.array([element[0] for element in dataset_train])
    y_train = np.array([element[1] for element in dataset_train])

    # Access the features and label from the test dataset
    X_test = np.array([element[0] for element in dataset_test])
    y_test = np.array([element[1] for element in dataset_test])

    return X_train, y_train, X_test, y_test

def remove_duplicates(dataset):

    dataset = list(dataset.as_numpy_iterator())
    unique_positions = set()
    unique_dataset = []
    for position, label in dataset:
        position_tuple = tuple(position.flatten())

        if position_tuple not in unique_positions:
            unique_positions.add(position_tuple)
            unique_dataset.append((position, label))
    return unique_dataset
    # # Save the unique dataset as a new TensorFlow database
    # unique_train_dataset = tf.data.Dataset.from_tensor_slices((X_unique, y_unique))
    # tf.data.experimental.save(unique_train_dataset, f'{save_path}_unique')

def filter_and_sample_labels(dataset, label_to_sample, sample_size):
    filtered_dataset = []
    count_label_to_sample = 0
    for idx, position_with_label in enumerate(dataset):
        # Extract position and label
        position = position_with_label[:-1]  # Remove the last element (label)
        label = position_with_label[-1]      # Extract the last element (label)

        if label == label_to_sample:
            count_label_to_sample += 1
            if count_label_to_sample <= sample_size:
                filtered_dataset.append((position, label))
        else:
            filtered_dataset.append((position, label))
    random.shuffle(filtered_dataset)

    return filtered_dataset, count_label_to_sample

global_count_deleted = 0

def filter_condition(element):
    global global_count_deleted

    if (np.abs(element[1]) - 10) >= 0 or global_count_deleted >= 300_000:
        global_count_deleted += 1
        return True
    else:
        return False


def serialize_example(input_data, label):
    input_bytes = tf.io.serialize_tensor(input_data)
    label_bytes = tf.io.serialize_tensor(label)

    feature = {
        'input': input_bytes.numpy(),
        'label': label_bytes.numpy()
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def filter_second_to_last_element_equal_to_1(x, y):
    second_to_last = x[:, :, -2][0][0]  # Assuming the second-to-last element is being checked
    condition = tf.equal(second_to_last, 1)
    if condition:
        return False
    return True

def main():
    # Load your data
    # y_train = np.load("y_train_big.npy")
    # X_train = tf.data.Dataset.load("saved_tf_train").unbatch()
    # y_train_unbatched = tf.data.Dataset.from_tensor_slices(y_train)

    # train_ds = tf.data.Dataset.zip((X_train, y_train_unbatched))

    # del y_train
    # del y_train_unbatched
    # del X_train

    # filtered_less_than_10_ds = train_ds.filter(lambda x, y: tf.abs(y) < 10)

    # filtered_greater_than_10_ds = train_ds.filter(lambda x, y: tf.abs(y) > 10)

    # filtered_less_than_10_sampled = filtered_less_than_10_ds.skip(300_000)
    # combined_dataset = filtered_less_than_10_sampled.concatenate(filtered_greater_than_10_ds).batch(512)
    # del filtered_greater_than_10_ds
    # del filtered_less_than_10_ds
    # gc.collect()
    # # ds_1 = combined_dataset.skip(1_000_000)

    # # tf.data.Dataset.save(ds_1, path="filtered_dataset_1")
    # # del ds_1
    # # tf.data.Dataset.save(ds_3, path="filtered_dataset_3")
    # print("AAAa")


    # tf.data.Dataset.save(combined_dataset, path="filtered_dataset")
    # print("hey")
    train_ds = tf.data.Dataset.load("filtered_dataset").unbatch()
    del train_ds
    gc.collect()
    y_test = np.load("y_test_big.npy")
    y_test = tf.data.Dataset.from_tensor_slices(y_test)
    X_test = tf.data.Dataset.load("saved_tf_test").unbatch()
    
    test_ds = tf.data.Dataset.zip((X_test, y_test))
    white_move_filtered = test_ds.filter(filter_second_to_last_element_equal_to_1)

    print("GAGAG")
    def data_generator(data):
        for x, y in data:
            last_dimension = x[:, :, -1]
            updated_last_dimension = last_dimension - 1
            updated_x = tf.concat([x[:, :, :-1], tf.expand_dims(updated_last_dimension, axis=-1)], axis=-1)
            yield (updated_x, y)

    data_generator = data_generator(white_move_filtered)

    del white_move_filtered
    gc.collect()
    ds = tf.data.Dataset.from_generator(lambda: data_generator, (tf.float32, tf.int32), (tf.TensorShape([None, None, None]), tf.TensorShape([])))
    tf.data.Dataset.save(ds, "new_test_ds")



if __name__ == "__main__":
    main()