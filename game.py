class game():
    """
    The abstract game class that all games must inherit from
    """

    def make_move(self,move):
        if move not in self.get_valid_moves():
            print('Invalid move attempted')
            return False

    def get_valid_moves(self):
        return []