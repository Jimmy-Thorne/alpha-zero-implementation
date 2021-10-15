from typing import Callable
from game import game
from player import player
from board import board

import numpy as np
import copy
import random

# At this point maybe for memory management there should be some pruning, not sure if that matters that much though
def minimax(game: game, board: board, depth, maximizing_player: bool, value_function: Callable[[game, board], float]):
    """
    The abstract minimax function takes in the current game it is running for, the current possible future board it's evaluating,
    whether or not this call is for the maximizing player,
    and a function which takes in a game (the current game state) and board (future board to evaluate) and returns a real number
    """
    valid_moves = board.get_valid_moves()
    if depth == 0 or game.check_for_finality(board):
        return value_function(game, board)
    if maximizing_player:
        value = -np.Inf
        for move in valid_moves:
            copy_board = copy.deepcopy(board)
            game.make_move(move, board)
            value = max(value, minimax(game, copy_board, depth - 1, False, value_function))
        return value
    else:
        value = np.Inf
        for move in valid_moves:
            copy_board = copy.deepcopy(board)
            game.make_move(move, board)
            value = min(value, minimax(game, copy_board, depth - 1, True, value_function))
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

    def make_move(self, game: game):
        """
        The abstract minimax algorithm.
        """

        valid_moves = game.get_valid_moves()

        best_moves = []
        best_value = -np.Inf

        for move in valid_moves:
            board = copy.deepcopy(game.board)
            game.make_move(move, board)
            move_value = minimax(game, board, self.depth - 1, False, self.value)
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