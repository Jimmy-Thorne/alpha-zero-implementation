import numpy as np

from game import game
from connect_4_board import connect_4_board
from player import player

class connect_4_game(game):
    """
    Class that inherits the functionality of game, specifically implements for Connect 4.
    """
    def __init__(self, black_player: player, red_player: player):
        """
        We initialize the game state to a 6 by 7 array of all 'O'.
        """
        self.board = connect_4_board()
        self.players = (black_player, red_player)
        self.black_player = black_player
        self.red_player = red_player
        self.current_player = black_player

    def check_for_finality(self, board: connect_4_board = None):
        if board is None:
            board = self.board
        
        # First check for horizontal wins
        for i in range(6):
            for j in range(4):
                if self.board.state[i,j] == 'O': continue # Certainly isn't a win
                if board.state[i,j] == board.state[i,j+1] == board.state[i,j+2] == board.state[i,j+3]:
                    if board.state[i,j] == 'R':
                        board.state_description = 'Red wins.'
                    else:
                        board.state_description = 'Black wins.'
                    return True
        
        # Second check for vertical wins
        for i in range(3):
            for j in range(7):
                if board.state[i,j] == 'O': continue # Certainly isn't a win
                if board.state[i,j] == board.state[i+1,j] == board.state[i+2,j] == board.state[i+3,j]:
                    if board.state[i,j] == 'R':
                        board.state_description = 'Red wins.'
                    else:
                        board.state_description = 'Black wins.'
                    return True
        
        # Third check for downwards diagonal wins
        for i in range(3):
            for j in range(4):
                if board.state[i,j] == 'O': continue # Certainly isn't a win
                if board.state[i,j] == board.state[i+1,j+1] == board.state[i+2,j+2] == board.state[i+3,j+3]:
                    if board.state[i,j] == 'R':
                        board.state_description = 'Red wins.'
                    else:
                        board.state_description = 'Black wins.'
                    return True

        # Fourth check for upwards diagonal wins
        for i in range(3,6):
            for j in range(4):
                if board.state[i,j] == 'O': continue # Certainly isn't a win
                if board.state[i,j] == board.state[i-1,j+1] == board.state[i-2,j+2] == board.state[i-3,j+3]:
                    if board.state[i,j] == 'R':
                        board.state_description = 'Red wins.'
                    else:
                        board.state_description = 'Black wins.'
                    return True

        # We have no connect 4's. Now check for a draw (no valid moves)
        if self.get_valid_moves(board) == []:
            board.state_description = 'The game is a draw.'
            return True
        
        # Seems like the game can keep going
        return False

    def get_valid_moves(self, board: connect_4_board = None):
        if board is None:
            board = self.board
        
        valid_moves = []

        for i in range(7):
            if board.state[0,i] == 'O': # Element in top of column
                valid_moves.append(board.current_player+str(i+1)) # +1 to convert back to humanspeak
        
        return valid_moves

    def make_move(self, move: str, board: connect_4_board = None):
        """
        Updates the game state to reflect the given move.
        Returns False if the move was not made and returns True if the move was made.
        """
        if board is None:
            board = self.board
        
        if move not in self.get_valid_moves(board):
            print("Invalid move, " + move + " attempted in position: ")
            self.show_game_state()
            return False
        for i in range(6):
            # The move[1]-1 is because we will let users assume columns 1 - 7 while we have 0 based indexing
            # The 5-i is so that we start at the bottom of the board
            if board.state[5-i,int(move[1])-1] == 'O':
                board.state[5-i,int(move[1])-1] = move[0] # Change the desired slot on board
                if board.current_player == 'R': # Update current player and state description
                    board.current_player = 'B'
                    board.state_description = 'Black to play.'
                else:
                    board.current_player = 'R'
                    board.state_description = 'Red to play.'
                return True # Return that it's a successful move
            if i == 5:
                print('End of column reached. Serious error.') # This should not be possible to reach
                return False 
    
    def play(self):
        return super().play()

    def show_game_state(self):
        print(self.board.state_description)
        print(self.board.state)