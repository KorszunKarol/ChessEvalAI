from dataclasses import dataclass, field
import multiprocessing
from multiprocessing import Queue
from config import Config
from pgn_processor import PGNProcessor
from position_evaluator import PositionEvaluator
import logging
import os
import time

@dataclass
class MultiprocessingCoordinator:
    config: Config
    target_size: int
    num_evaluators: int = field(default_factory=lambda: Config.NUM_PROCESSES)

    def run(self):
        logging.info(f"Starting MultiprocessingCoordinator with target size: {self.target_size} and {self.num_evaluators} evaluators")
        input_queue = Queue()
        output_queue = Queue()
        stop_event = multiprocessing.Event()

        evaluators = []
        for i in range(self.num_evaluators):
            evaluator = PositionEvaluator(self.config, input_queue, output_queue, stop_event)
            process = multiprocessing.Process(target=evaluator.run)
            process.start()
            evaluators.append(process)
            logging.info(f"Started evaluator process {i+1} with PID: {process.pid}")

        pgn_processor = PGNProcessor(self.config.PGN_FILE_PATH, chunk_size=1_000_000_000, min_move_number=self.config.MIN_MOVE_NUMBER)
        position_count = 0
        matrices = []
        scores = []

        logging.info("Beginning to process games")
        try:
            for fen, matrix in pgn_processor.process_games():
                try:
                    input_queue.put((fen, matrix), timeout=5)
                    position_count += 1
                    if position_count % 100 == 0:  # Log every 100 positions
                        logging.info(f"Processed {position_count} positions")
                except multiprocessing.queues.Full:
                    logging.error("Input queue is full. Unable to enqueue new positions.")
                    break

                if position_count >= self.target_size:
                    logging.info(f"Reached target size of {self.target_size} positions.")
                    break

            logging.info(f"Finished processing games. Processed {position_count} positions in total.")
            logging.info("Waiting for evaluations to complete...")

            # Allow time for evaluations to complete
            time.sleep(10)

            logging.info("Collecting results.")
            while len(matrices) < position_count:
                try:
                    matrix, score = output_queue.get(timeout=1)
                    matrices.append(matrix)
                    scores.append(score)
                    if len(matrices) % 100 == 0:  # Log every 100 collected results
                        logging.info(f"Collected {len(matrices)} results")
                except multiprocessing.queues.Empty:
                    if len(matrices) == position_count:
                        break
                    logging.warning("Timeout while waiting for output queue. Retrying...")
                    continue

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)

        finally:
            logging.info("Initiating shutdown process")
            # Signal all evaluator processes to stop
            stop_event.set()
            logging.info("Signaling evaluator processes to terminate.")
            for i, evaluator in enumerate(evaluators):
                evaluator.join(timeout=10)
                if evaluator.is_alive():
                    logging.warning(f"Process {evaluator.pid} (evaluator {i+1}) did not terminate gracefully and will be forcefully terminated.")
                    evaluator.terminate()
                else:
                    logging.info(f"Process {evaluator.pid} (evaluator {i+1}) terminated gracefully")

            # Cleanup queues
            logging.info("Cleaning up queues")
            input_queue.close()
            output_queue.close()
            input_queue.join_thread()
            output_queue.join_thread()

            logging.info(f"Data extraction pipeline has been terminated. Collected {len(matrices)} positions out of {position_count} processed.")

        return matrices, scores