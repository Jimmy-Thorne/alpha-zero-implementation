from board import board

class player():
    """
    The abstract player class that all players must inherit from.
    """

    def __init__(self, name: str = None):
        self.player_name = name

    def choose_move(self, board: board) -> str:
        """
        The logic for choosing a move on a board.
        """
        return None