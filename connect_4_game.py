import numpy as np

from game import game
from connect_4_board import connect_4_board
from player import player

class connect_4_game(game):
    """
    Class that inherits the functionality of game, specifically implements for Connect 4.
    """
    def __init__(self, black_player: player, red_player: player):
        self.board = connect_4_board()
        self.players = (black_player, red_player)
        black_player.key = 'B'
        red_player.key = 'R'
        self.current_player = black_player

    def play(self, show_state_on_finality = False):
        return super().play(show_state_on_finality)