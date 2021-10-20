from typing import Callable
from player import player
from board import board

import numpy as np
import copy
import random

# At this point maybe for memory management there should be some pruning, not sure if that matters that much though
def minimax(root_board: board, current_board: board, depth, maximizing_player: bool, value_function: Callable[[board, board], float]):
    """
    The abstract minimax function. It wants to know the root board its perspective is from,
    and the current board it is analyzing.
    """

    # First check if we have reached our desired minimax depth or if the boardstate is a final state.
    if depth == 0 or current_board.check_for_finality():
        return value_function(root_board, current_board)

    # Now is the logic for the minimax algorithm
    valid_moves = current_board.get_valid_moves()
    if maximizing_player:
        value = -np.Inf
        for move in valid_moves:
            copy_board = copy.deepcopy(current_board)
            copy_board.make_move(move)
            value = max(value, minimax(root_board, copy_board, depth - 1, False, value_function))
        return value
    else:
        value = np.Inf
        for move in valid_moves:
            copy_board = copy.deepcopy(current_board)
            copy_board.make_move(move)
            value = min(value, minimax(root_board, copy_board, depth - 1, True, value_function))
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

    def choose_move(self, board: board) -> str:
        """
        The abstract minimax algorithm.
        """

        valid_moves = board.get_valid_moves()

        best_moves = []
        best_value = -np.Inf

        for move in valid_moves:
            copy_board = copy.deepcopy(board)
            copy_board.make_move(move)
            move_value = minimax(board, copy_board, self.depth - 1, False, self.value)
            if move_value == best_value:
                best_moves.append(move)
            elif move_value > best_value:
                best_moves = [move]
                best_value = move_value
            
        best_move = best_moves[random.randrange(len(best_moves))]
        
        return best_move

    def value(self, root_board: board, current_board: board) -> float:
        """
        The value of the current_board from the perspective of the root_board
        """
        return 0