from dataclasses import dataclass, field
import chess.engine
import os
import logging
from config import Config

@dataclass
class EngineManager:
    config: Config
    engine: chess.engine.SimpleEngine = field(init=False)

    def __post_init__(self):
        self.engine_path = os.path.join(self.config.STOCKFISH_DIRECTORY, self.config.STOCKFISH_FILE)
        logging.info(f"Initializing Stockfish engine at: {self.engine_path}")

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            logging.info("Stockfish engine initialized successfully.")
        except chess.engine.EngineTerminatedError as e:
            logging.error(f"Engine terminated unexpectedly: {e}")
            raise
        except PermissionError as e:
            logging.error(f"Permission error when initializing engine: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error when initializing engine: {e}")
            raise

    def analyze(self, fen: str, time_limit: float = 0.1) -> int:
        try:
            board = chess.Board(fen)
            info = self.engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = info["score"].relative

            if score.is_mate():
                # If it's a mate score, return 2000 or -2000 based on whether it's in favor of white or black
                return 2000 if score.mate() > 0 else -2000
            else:
                # For non-mate scores, return the centipawn value
                return score.score()
        except Exception as e:
            logging.error(f"Error during position analysis: {e}")
            return 0  # Return 0 for any error, indicating an inconclusive evaluation

    def close(self):
        if hasattr(self, 'engine'):
            self.engine.quit()
            logging.info("Stockfish engine closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()