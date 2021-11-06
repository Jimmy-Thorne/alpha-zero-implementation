from abc import abstractmethod
from board import board
from player import player

class nn():
    def __init__(self, checkpoint_name: str = None) -> None:
        return

    def __call__(self, board: board, player: player) -> float:
        return

    def save(self) -> None:
        return

    def train(self) -> None:
        return