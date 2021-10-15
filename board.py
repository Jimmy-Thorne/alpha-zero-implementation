class board():
    """
    Meant to keep track of information about a possible game state without making new instances of a game.
    The abstract class which all other boards must inherit from.
    """

    def __init__(self):
        self.current_player = None
        self.state = None
        self.state_description = None

    def get_valid_moves(self):
        return []

    def make_move(self, move: str):
        return False