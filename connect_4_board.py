from board import board

import numpy as np

class connect_4_board(board):
    """
    Keeps track of relevant data to the state of a connect 4 board
    """
    def __init__(self):
        self.state = np.full((6,7),'O')
        self.state_description = 'Black to play.'
        self.current_player = 'B'
