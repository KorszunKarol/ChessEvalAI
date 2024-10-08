from dataclasses import dataclass
from multiprocessing import Queue, Event
from engine_manager import EngineManager
from config import Config
import logging
import os
import gc
import psutil

@dataclass
class PositionEvaluator:
    config: Config
    input_queue: Queue
    output_queue: Queue
    stop_event: Event
    max_memory_percent: float = 80.0  # Maximum memory usage percentage

    def run(self):
        logging.info(f"PositionEvaluator starting. Process ID: {os.getpid()}")
        try:
            with EngineManager(self.config) as engine_manager:
                logging.info("EngineManager initialized successfully")
                while not self.stop_event.is_set():
                    try:
                        if self.check_memory_usage():
                            fen, matrix = self.input_queue.get(timeout=1)
                            score = engine_manager.analyze(fen)

                            self.output_queue.put((matrix, score))
                            logging.debug(f"Evaluated position. FEN: {fen}, Score: {score}")

                            # Explicitly delete objects to free memory
                            del fen, matrix, score
                            gc.collect()
                        else:
                            logging.warning("Memory usage too high. Waiting for memory to be freed.")
                            self.wait_for_memory()
                    except Queue.Empty:
                        continue
                    except Exception as e:
                        logging.error(f"Error during position evaluation: {e}")
        except Exception as e:
            logging.error(f"Error initializing or using EngineManager: {e}")
        finally:
            logging.info("PositionEvaluator shutting down")

    def check_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        return memory_percent < self.max_memory_percent

    def wait_for_memory(self, wait_time=1):
        import time
        time.sleep(wait_time)
        gc.collect()