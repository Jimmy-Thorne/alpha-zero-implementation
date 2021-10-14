from player import player
from game import game
import random

class random_player(player):
    """
    A player who always picks a random move.
    """

    def __init__(self):
        self.player_name = 'Random player'

    def make_move(self, game):
        valid_moves = game.get_valid_moves()

        num_moves = len(valid_moves)

        game.make_move(valid_moves[random.randrange(num_moves)])
