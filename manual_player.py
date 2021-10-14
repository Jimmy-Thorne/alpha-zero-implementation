from player import player
from game import game

class manual_player(player):
    """
    A player which choses it's moves via input from a user.
    """

    def __init__(self, name):
        self.player_name = name

    def make_move(self, game):
        """
        A manual player must input a move into the console.
        """

        valid_moves = game.get_valid_moves() # Just do this once now

        game.show_game_state() # Show the state on the console
        print('Valid moves: ' + str(valid_moves))

        chosen_move = ''
        print('Please input a valid move')
        while chosen_move not in valid_moves:
            chosen_move = input()
            if chosen_move not in valid_moves:
                print('Invalid move, try again...')

        game.make_move(chosen_move)