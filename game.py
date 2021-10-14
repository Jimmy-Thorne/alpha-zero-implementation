class game():
    """
    The abstract game class that all games must inherit from.
    """

    def __init__(self):
        self.game_state = None
        self.state_description = None
        self.current_player = None
        self.valid_moves = None

    def check_for_finality(self):
        return False

    def get_game_state(self):
        return None

    def get_valid_moves(self):
        return []

    def make_move(self,move):
        return False

    def show_game_state(self):
        pass