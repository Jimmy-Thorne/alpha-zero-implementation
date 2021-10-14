from game import game
from player import player

import numpy as np
import copy
import random

# We really shouldn't have to make all these copys of the game
# The structure of the game class should probably be remade to allow for easier execution of game trees
def minimax(game, depth, maximizing_player, value_function):
    valid_moves = game.get_valid_moves()
    if depth == 0 or game.check_for_finality():
        return value_function(game)
    if maximizing_player:
        value = -np.Inf
        for move in valid_moves:
            copy_game = copy.deepcopy(game)
            copy_game.make_move(move)
            value = max(value, minimax(copy_game, depth - 1, False, value_function))
        return value
    else:
        value = np.Inf
        for move in valid_moves:
            copy_game = copy.deepcopy(game)
            copy_game.make_move(move)
            value = min(value, minimax(copy_game, depth - 1, True, value_function))
        return value

class minimax_player(player):
    """
    An abstract class of a player who implements the minimax algorithm to make decisions.
    """

    def __init__(self, depth = 1, name = 'Minimax player'):
        """
        A minimax player can define different search depths
        """
        self.player_name = name
        self.depth = depth

    def make_move(self, game):
        """
        The abstract minimax algorithm.
        """

        valid_moves = game.get_valid_moves()

        best_moves = []
        best_value = -np.Inf

        for move in valid_moves:
            copy_game = copy.deepcopy(game) # Again, we should find a way to change this
            copy_game.make_move(move)
            move_value = minimax(copy_game, self.depth - 1, False, self.value)
            if move_value == best_value:
                best_moves.append(move)
            elif move_value > best_value:
                best_moves = [move]
                best_value = move_value
            
        best_move = best_moves[random.randrange(len(best_moves))]
        
        game.make_move(best_move)

    def value(self, game):
        """
        This needs to be defined in child classes.
        """
        return 0