from minimax_player import minimax_player
from connect_4_board import connect_4_board
from connect_4_game import connect_4_game

class connect_4_minimax_player(minimax_player):
    """
    The class implementing the minimax value function for the connect 4 game
    """
    def make_move(self, game):
        return super().make_move(game)

    def value(self, game: connect_4_game, board: connect_4_board):
        if game.check_for_finality(board):
            if game.black_player == self:
                if board.state_description == 'Black wins.':
                    return 1
                elif board.state_description == 'Red wins.':
                    return -1
                else:
                    return 0
            else:
                if board.state_description == 'Black wins.':
                    return -1
                elif board.state_description == 'Red wins.':
                    return 1
                else:
                    return 0
        return 0