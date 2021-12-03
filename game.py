from board import board
from player import player

class game():
    """
    The abstract game class that all games must inherit from.
    """

    def __init__(self, *players: player):
        self.current_player = players[0] # The first player to go will be the first player in the list
        self.board = board()
        self.players = players

    def play(self, show_board_on_finality = False) -> None:
        while True:
            # First the current player will select a move for the current board.
            move = self.current_player.choose_move(self.board)

            # board.make_move(move) will always run.
            # If it returns false we enter the if statement and loop back up, not changing players.
            # as an invalid move has been made.
            if not self.board.make_move(move):
                print('Player ' + self.current_player.player_name + ' attempted invalid move: ' + move + ' in position')
                self.board.show()
                continue

            # Check to see if the game is over.
            if self.board.check_for_finality():
                # If we want to display the final board state, do that.
                if show_board_on_finality:
                    self.board.show()
                break # Stop playing the game.

            # If a valid move was made and the game is not over, move to the next player.
            if self.current_player == self.players[-1]:
                self.current_player = self.players[0]
            else:
                self.current_player = self.players[self.players.index(self.current_player) + 1]
    
    def play_many(self, rounds = 10):
        dict = {}

        for player in self.players:
            dict[str(player)] = 0
        dict['Draw'] = 0

        for i in range(rounds):
            print('Game {0}/{1}       '.format(i+1, rounds), end='\r')
            self.current_player = self.players[i % len(self.players)]
            self.board.reset()

            self.play()

            if self.board.winner is None:
                dict['Draw'] += 1
            else:
                for player in self.players:
                    if player.key == self.board.winner:
                        dict[str(player)] += 1
        
        print('--- play_many report ---')
        for player in self.players:
            print('\t{0}: {1}'.format(str(player), dict[str(player)]))
        print('\tDraws: {0}'.format(dict['Draw']))

