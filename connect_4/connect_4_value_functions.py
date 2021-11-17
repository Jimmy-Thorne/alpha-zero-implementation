"""
A connect 4 value function is a function which takes in a connect_4_board and a player and returns a float in [-1,1]
The closer to -1 the more likely player is to lose.
The closer to +1 the more likely player is to win.
"""

from connect_4.connect_4_board import connect_4_board
from player import player

def naive(board: connect_4_board, player: player):
    """
    The naive value function only checks to see if the game is over and see if the player won.
    """
    if board.check_for_finality():
        if player.key == 'B':
            if board.description == 'Black wins.': return 1 
            elif board.description == 'Red wins.': return -1
            else: return 0
        else:
            if board.description == 'Black wins.': return -1
            elif board.description == 'Red wins.': return 1
            else: return 0
    return 0