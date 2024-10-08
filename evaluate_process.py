import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


dataset_path = '/home/karolito/DL/chess/big_dataset'

# Load the dataset
dataset = tf.data.experimental.load(dataset_path)

# Define a function to extract the label from each element
def extract_label(position, label):
    return label

# Use the map function to extract labels efficiently
labels_dataset = dataset.map(extract_label)

# Convert the labels dataset to a NumPy array
labels = np.array(list(labels_dataset.as_numpy_iterator()))

# Now `labels` contains all the labels from the dataset
print(labels)
plt.figure(figsize=(12, 6))

x_axis_range = (-1000, 1000)
num_bins = 200  # Increase the number of bins to 200 or adjust as needed

plt.hist(labels, bins=num_bins, range=x_axis_range, edgecolor='black')  # Plot the histogram
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Label Distribution Histogram')
plt.xticks(np.arange(x_axis_range[0], x_axis_range[1]+1, 100))  # Adjust the x-axis ticks to show more granularity
plt.grid(axis='y', linestyle='dotted', alpha=0.7)  # Add grid lines for better readability

plt.show()
