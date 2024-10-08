import tensorflow as tf
import portion
import numpy as np



def get_label_value(label):
    abs_label = np.abs(label)
    dict = portion.IntervalDict()
    dict[portion.closed(-0.3, 0.3)] = 0
    dict[portion.closed(0.3, 0.8)] = 1
    dict[portion.closed(0.8, 1.4)] = 2
    dict[portion.closed(1.4, 1.9)] = 3
    dict[portion.closed(1.9, 2.4)] = 4
    dict[portion.closed(2.4, 2.9)] = 5
    dict[portion.closed(2.9, 4)] = 6
    dict[portion.closed(4, 5)] = 7
    dict[portion.closed(5, 7)] = 8
    dict[portion.closed(7, float('inf'))] = 9

    if label >= 0:
        return dict[abs_label]
    else:
        return -dict[abs_label]


def extract_and_process_labels(path):
    dataset = tf.data.Dataset.load(path)
    labels = []
    features = []
    for ix, element in enumerate(dataset):
        if ix > 750_000:
            print(ix)
            label_value = get_label_value(float(element[1]) / 100)
            labels.append(label_value)
            features.append(element[0])
        else:
            print(ix)
            continue
    return (features, labels)


def main():
    path = "/home/karolito/DL/chess/filtered_dataset"
    save_path = "/home/karolito/DL/chess/labeled_dataset_2"
    data = extract_and_process_labels(path)
    labeled_dataset = tf.data.Dataset.from_tensor_slices(data)
    tf.data.Dataset.save(labeled_dataset, save_path)


if __name__ == "__main__":
    main()