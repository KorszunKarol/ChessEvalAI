from dataclasses import dataclass
import os
import logging

@dataclass
class Config:
    STOCKFISH_DIRECTORY: str = "/home/karolito/DL/chess/Stockfish/src"
    STOCKFISH_FILE: str = "stockfish"
    PGN_FILE_PATH: str = r"/mnt/d/lichess_db_standard_rated_2024-08.pgn"
    DATA_FILE_PATH: str = "data_test_new.txt"
    SAVE_PATH: str = os.path.join(os.getcwd(), 'preprocessed_data')
    NUM_PROCESSES: int = 12
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_LEVEL: int = logging.INFO
    MIN_MOVE_NUMBER: int = 8

    def __post_init__(self):
        logging.basicConfig(level=self.LOG_LEVEL, format=self.LOG_FORMAT)
        logging.getLogger('chess.engine').setLevel(logging.WARNING)
        self.check_stockfish()

        # Add this line to use system libraries instead of Anaconda's
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

    def check_stockfish(self):
        stockfish_path = os.path.join(self.STOCKFISH_DIRECTORY, self.STOCKFISH_FILE)
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish binary not found at {stockfish_path}")
        if not os.access(stockfish_path, os.X_OK):
            raise PermissionError(f"Stockfish binary at {stockfish_path} is not executable")
        logging.info(f"Stockfish binary found at {stockfish_path}")

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)