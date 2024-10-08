import tensorflow as tf
from model import convert_array
import numpy as np
import os



path_1= '/home/karolito/DL/chess/saved_tf_test'
dataset1 = tf.data.Dataset.load(path_1)
length = len(dataset1)


ds1 = np.load("y_test.npy")
ds2 = np.load("y_test_600k.npy")
result = np.concatenate((ds1, ds2), axis=0)
np.save("y_test_big.npy", result)
print(len(result))

# for i in dataset1.unbatch():
#     print(i)




# path_2 = '/home/karolito/DL/chess/18.12.New_data_600k'
# dataset2 = tf.data.Dataset.load(path_2)

# # # Merge the datasets
# merged_dataset = dataset1.concatenate(dataset2)
# save_path = '/home/karolito/DL/chess/big_dataset'
# tf.data.experimental.save(merged_dataset, save_path)



