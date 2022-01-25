class board():
    """
    Meant to keep track of information about a possible game state without making new instances of a game.
    The abstract class which all other boards must inherit from.
    """

    def __init__(self):
        self.current_player = None
        self.state = None
        self.description = None
        self.winner = None

    def check_for_finality(self) -> bool:
        return False

    def get_valid_moves(self) -> 'list[str]':
        return []
    
    def make_move(self, move: str) -> bool:
        return False

    def show(self) -> None:
        pass

    def reset(self) -> None:
        pass