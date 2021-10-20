from minimax_player import minimax_player
from connect_4_board import connect_4_board

class connect_4_minimax_player(minimax_player):
    """
    The class implementing the minimax value function for the connect 4 game
    """
    def choose_move(self, board: connect_4_board):
        return super().choose_move(board)

    def value(self, root_board: connect_4_board, current_board: connect_4_board):
        if current_board.check_for_finality():
            if root_board.current_player == 'B':
                if current_board.description == 'Black wins.':
                    return 1
                elif current_board.description == 'Red wins.':
                    return -1
                else:
                    return 0
            else:
                if current_board.description == 'Black wins.':
                    return -1
                elif current_board.description == 'Red wins.':
                    return 1
                else:
                    return 0
        return 0