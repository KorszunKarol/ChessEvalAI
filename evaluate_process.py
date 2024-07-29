import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


dataset_path = '/home/karolito/DL/chess/big_dataset'
dataset = tf.data.experimental.load(dataset_path)

def extract_label(position, label):
    return label

labels_dataset = dataset.map(extract_label)

labels = np.array(list(labels_dataset.as_numpy_iterator()))

print(labels)
plt.figure(figsize=(12, 6))

x_axis_range = (-1000, 1000)
num_bins = 200 

plt.hist(labels, bins=num_bins, range=x_axis_range, edgecolor='black')  
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Label Distribution Histogram')
plt.xticks(np.arange(x_axis_range[0], x_axis_range[1]+1, 100))  
plt.grid(axis='y', linestyle='dotted', alpha=0.7)  
plt.show()
