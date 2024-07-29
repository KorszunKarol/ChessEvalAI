# Chess Position Evaluation AI

## Overview

This project is a Chess Position Evaluation AI that predicts the evaluation of a chess position in centipawns. The AI was trained using supervised learning on approximately 2 million chess positions from the Lichess open chess database. These positions were evaluated using Stockfish to create labels for the dataset.

## Features

- Predicts chess position evaluations in centipawns
- Utilizes a deep learning model trained on a large dataset of chess positions
- Implements various custom loss functions and neural network architectures
- Includes data processing and augmentation techniques

## Project Structure

The project consists of three main Python scripts:

1. `database.py`: Handles data processing, including FEN to matrix conversion and material score calculation.
2. `model.py`: Contains the neural network model definition, custom loss functions, and training loop.
3. `search.py`: Implements a chess engine search algorithm using the trained model and alpha-beta search.

## Model Architecture

The neural network model uses a combination of convolutional and dense layers, inspired by the AlphaZero approach. It includes:

- Multiple convolutional layers for feature extraction
- Batch normalization and activation layers
- Dense layers for final evaluation
- Custom loss function (weighted MSE) to prioritize certain types of positions

The input representation is similar to that used in the AlphaZero paper, where the chess position is represented as a stack of binary feature planes (bitmaps). This allows the convolutional layers to effectively process the spatial relationships between pieces on the board.

## Performance Optimization

To improve inference speed, the model was optimized using TensorRT. This optimization allows for faster position evaluation during the search process, enabling the engine to explore more positions in less time.


## License

This project is open source and available under the [MIT License](LICENSE).
