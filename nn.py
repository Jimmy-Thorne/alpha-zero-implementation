from typing import Callable
from board import board
from player import player
import numpy as np
import tensorflow as tf

class nn():
    """
    This class wraps a tensorflow neural network which evaluates a board, and can be trained.
    The reason this exists is so that we can call networks on (board,player) tuples directly.
    When initializing this class you tell it how to do that. Preparing a board to be handed to tensorflow can be annoying.
    """
    def __init__(self, model: tf.keras.Model, board_to_array: Callable[[board], np.array], name: str = None) -> None:
        self.model = model
        self.board_to_array = board_to_array
        self.name = name

    def __call__(self, board: board, player: player) -> float:
        result = self.model(np.array([self.board_to_array(board)]))
        if board.current_player == player.key:
            return result
        else:
            return result * -1

    def load(self, model_location: str) -> None:
        self.model = tf.keras.models.load_model(model_location)

    def save(self, model_location: str) -> None:
        self.model.save(model_location)