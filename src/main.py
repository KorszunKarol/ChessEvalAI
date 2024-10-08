from config.config import Config
from coordinator.multiprocessing_coordinator import MultiprocessingCoordinator

def main():
    # Initialize Configuration
    config = Config()

    # Log the start of the pipeline
    logging.info("Starting data extraction pipeline")

    # Initialize the Multiprocessing Coordinator with the desired target size
    coordinator = MultiprocessingCoordinator(config=config, target_size=20_000)

    # ... existing code ...

    # Save the dataset
    tf.data.experimental.save(dataset, config.SAVE_PATH)
    logging.info(f"Dataset saved to {config.SAVE_PATH}")

if __name__ == '__main__':
    main()