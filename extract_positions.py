import logging
import time
import tensorflow as tf
import numpy as np
from config import Config
from multiprocessing_coordinator import MultiprocessingCoordinator
from pathlib import Path
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cpu = tf.config.list_physical_devices('CPU')
    print(f"CPU devices: {cpu}")
    if cpu:
        tf.config.set_visible_devices(cpu, 'CPU')
        print("Running on CPU")
    else:
        print("No CPU devices found")
    config = Config()
    logging.info("Starting data extraction pipeline")
    np.set_printoptions(threshold=np.inf)
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        tf.config.set_visible_devices(physical_devices, 'CPU')
        print("Running on CPU")
    else:
        print("No CPU devices found")

    np.set_printoptions(threshold=np.inf)  # Use np.inf to display all elements

    pgn_path = Config.PGN_FILE_PATH
    print(f"PGN file exists: {os.path.exists(pgn_path)}")

    config.MIN_MOVE_NUMBER = 12  # Set the minimum move number

    batch_size = 100_000
    batch_number = 0

    while True:
        coordinator = MultiprocessingCoordinator(config, target_size=batch_size)

        tic = time.time()
        matrices, scores = coordinator.run()
        tac = time.time()

        logging.info(f"Processed {len(matrices)} positions in {tac - tic:.2f} seconds")

        if len(matrices) == 0:
            logging.warning("No positions processed. Exiting.")
            break

        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((matrices, scores))

        # Create a unique filename for this batch
        timestamp = int(time.time())
        save_path = os.path.join(config.SAVE_PATH, f"dataset_batch_{batch_number}_{timestamp}")

        # Save the dataset
        tf.data.Dataset.save(dataset, save_path)
        logging.info(f"Dataset saved to {save_path}")

        # Verify the saved dataset
        loaded_dataset = tf.data.experimental.load(save_path)
        for x, y in loaded_dataset.take(1):
            print(f"Shape of x in batch {batch_number}:", x.shape)
            print(f"Shape of y in batch {batch_number}:", y.shape)

        batch_number += 1
        logging.info(f"Completed batch {batch_number}. Starting next batch...")

if __name__ == '__main__':
    main()