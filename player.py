class Player():
    """docstring for Player."""

    def __init__(self):
        super(Player, self).__init__()

    def get_move(self,valid_moves):
        x = input()
        while x not in valid_moves:
            print("Invalid Move, please try again")
            x = input()
        return x

    def choose_move(self):
        #valid_moves = game.valid_moves() # TODO: Match up with Nick
        valid_moves = ['2','3','4']
        print("The current valid moves are ")
        print(valid_moves)
        return self.get_move(valid_moves)




p1 = Player()
print(p1.choose_move())
