from typing import Callable
from player import player
from board import board

import numpy as np
import copy
import random

# At this point maybe for memory management there should be some pruning, not sure if that matters that much though
def minimax(current_board: board, player: player, depth, maximizing_player: bool, value_function: Callable[[board, player], float]):
    """
    The abstract minimax function. It wants to know who is asking and what board they are asking about.
    """

    # First check if we have reached our desired minimax depth or if the boardstate is a final state.
    if depth == 0 or current_board.check_for_finality():
        return value_function(current_board, player)

    # Now is the logic for the minimax algorithm
    valid_moves = current_board.get_valid_moves()
    if maximizing_player:
        value = -np.Inf
        for move in valid_moves:
            copy_board = copy.deepcopy(current_board)
            copy_board.make_move(move)
            value = max(value, minimax(copy_board, player, depth - 1, False, value_function))
        return value
    else:
        value = np.Inf
        for move in valid_moves:
            copy_board = copy.deepcopy(current_board)
            copy_board.make_move(move)
            value = min(value, minimax(copy_board, player, depth - 1, True, value_function))
        return value

class minimax_player(player):
    """
    An abstract class of a player who implements the minimax algorithm to make decisions.
    """

    def __init__(self, value_function: Callable[[board, player], float], depth = 1, name = 'Minimax player', key = None):
        """
        A minimax player can define different search depths and different value functions.
        """
        self.name = name
        self.depth = depth
        self.key = key
        self.value_function = value_function

    def choose_move(self, board: board) -> str:
        """
        The minimax player runs the minimax algorithm and returns it's chosen move.
        """

        valid_moves = board.get_valid_moves()

        best_moves = []
        best_value = -np.Inf

        for move in valid_moves:
            copy_board = copy.deepcopy(board)
            copy_board.make_move(move)
            move_value = minimax(copy_board, self, self.depth - 1, False, self.value_function)
            if move_value == best_value:
                best_moves.append(move)
            elif move_value > best_value:
                best_moves = [move]
                best_value = move_value
            
        return best_moves[random.randrange(len(best_moves))]