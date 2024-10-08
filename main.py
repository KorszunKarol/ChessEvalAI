import logging
import time
import tensorflow as tf
from config import Config
from multiprocessing_coordinator import MultiprocessingCoordinator

def main():
    logging.info("Starting data extraction pipeline")

    coordinator = MultiprocessingCoordinator(Config, target_size=20_000)

    tic = time.time()
    matrices, scores = coordinator.run()
    tac = time.time()

    logging.info(f"Processed {len(matrices)} positions in {tac - tic} seconds")

    dataset = tf.data.Dataset.from_tensor_slices((matrices, scores))
    tf.data.experimental.save(dataset, Config.SAVE_PATH)
    logging.info(f"Dataset saved to {Config.SAVE_PATH}")

if __name__ == '__main__':
    main()