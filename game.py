from board import board
from player import player

class game():
    """
    The abstract game class that all games must inherit from.
    """

    def __init__(self, *players: player):
        self.current_player = players[0]
        self.board = board()
        self.players = players

    def check_for_finality(self, board: board = None):
        if board is None:
            board = self.board
        return False

    def get_valid_moves(self, board: board = None):
        if board is None:
            board = self.board
        return []

    def make_move(self, move: str, board: board = None):
        if board is None:
            board = self.board
        return False

    def play(self):
        while True:   
            self.current_player.make_move(self)
            if self.check_for_finality(): break
            if self.current_player == self.players[-1]:
                self.current_player = self.players[0]
            else:
                self.current_player = self.players[self.players.index(self.current_player)+1]

    def show_board(self, board: board = None):
        if board is None:
            board = self.board
        pass