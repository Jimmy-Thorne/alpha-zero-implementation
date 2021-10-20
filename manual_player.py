from player import player
from board import board

class manual_player(player):
    """
    A player which choses it's moves via input from a user.
    """

    def __init__(self, name):
        self.player_name = name

    def choose_move(self, board: board) -> str:
        """
        A manual player must input a move into the console.
        """

        valid_moves = board.get_valid_moves() # Just do this once now

        board.show_game_state() # Show the state on the console
        print('Valid moves: ' + str(valid_moves))

        chosen_move = ''
        print('Please input a valid move')
        while chosen_move not in valid_moves:
            chosen_move = input()
            if chosen_move not in valid_moves:
                print('Invalid move, try again...')

        return chosen_move