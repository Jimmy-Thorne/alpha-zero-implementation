from player import player
from board import board
import random

class random_player(player):
    """
    A player who always picks a random move.
    """

    def __init__(self):
        self.player_name = 'Random player'

    def choose_move(self, board: board) -> str:
        valid_moves = board.get_valid_moves()

        num_moves = len(valid_moves)

        return valid_moves[random.randrange(num_moves)]
