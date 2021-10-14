from minimax_player import minimax_player

class connect_4_minimax_player(minimax_player):
    """
    The class implementing the minimax value function for the connect 4 game
    """
    def make_move(self, game):
        return super().make_move(game)

    def value(self, game):
        if game.check_for_finality():
            if game.black_player == self:
                if game.state_description == 'Black wins.':
                    return 1
                elif game.state_desciption == 'Red wins.':
                    return -1
                else:
                    return 0
            else:
                if game.state_description == 'Black wins.':
                    return -1
                elif game.state_description == 'Red wins.':
                    return 1
                else:
                    return 0
        return 0