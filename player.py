class player():
    """
    The abstract player class that all players must inherit from.
    """

    def __init__(self):
        self.player_name = ''

    def make_move(self, game):
        """
        The logic for choosing a move, and then calling the game to make the chosen move.
        """
        pass