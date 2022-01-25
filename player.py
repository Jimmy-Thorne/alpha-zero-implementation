from board import board

class player():
    """
    The abstract player class that all players must inherit from.
    """

    def __init__(self, name: str = None, key: str = None):
        self.name = name # A name for the player
        self.key = key # A unique key for the player. Usually the color of pieces they control.
    
    def __str__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def choose_move(self, board: board) -> str:
        """
        The logic for choosing a move on a board.
        """
<<<<<<< HEAD
        return
=======
        return
>>>>>>> edfc6e3d32e3698591983e51e9c240679150a5f9
