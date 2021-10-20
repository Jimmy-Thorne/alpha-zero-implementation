from board import board

import numpy as np

class connect_4_board(board):
    """
    Keeps track of relevant data to the state of a connect 4 board
    """
    def __init__(self):
        self.state = np.full((6,7),'O') # The board starts as all open spots
        self.description = 'Black to play.' # The board starts in this state
        self.current_player = 'B' # As black is the first player they are current after initialization

    def check_for_finality(self):
        # First check for horizontal wins
        for i in range(6):
            for j in range(4):
                if self.state[i,j] == 'O': continue # Certainly isn't a win
                if self.state[i,j] == self.state[i,j+1] == self.state[i,j+2] == self.state[i,j+3]:
                    if self.state[i,j] == 'R':
                        self.description = 'Red wins.'
                    else:
                        self.description = 'Black wins.'
                    return True
        
        # Second check for vertical wins
        for i in range(3):
            for j in range(7):
                if self.state[i,j] == 'O': continue # Certainly isn't a win
                if self.state[i,j] == self.state[i+1,j] == self.state[i+2,j] == self.state[i+3,j]:
                    if self.state[i,j] == 'R':
                        self.description = 'Red wins.'
                    else:
                        self.description = 'Black wins.'
                    return True
        
        # Third check for downwards diagonal wins
        for i in range(3):
            for j in range(4):
                if self.state[i,j] == 'O': continue # Certainly isn't a win
                if self.state[i,j] == self.state[i+1,j+1] == self.state[i+2,j+2] == self.state[i+3,j+3]:
                    if self.state[i,j] == 'R':
                        self.description = 'Red wins.'
                    else:
                        self.description = 'Black wins.'
                    return True

        # Fourth check for upwards diagonal wins
        for i in range(3,6):
            for j in range(4):
                if self.state[i,j] == 'O': continue # Certainly isn't a win
                if self.state[i,j] == self.state[i-1,j+1] == self.state[i-2,j+2] == self.state[i-3,j+3]:
                    if self.state[i,j] == 'R':
                        self.description = 'Red wins.'
                    else:
                        self.description = 'Black wins.'
                    return True

        # We have no connect 4's. Now check for a draw (no valid moves)
        if self.get_valid_moves() == []:
            self.description = 'The game is a draw.'
            return True
        
        # Seems like the game can keep going
        return False

    def get_valid_moves(self):
        valid_moves = []

        for i in range(7):
            if self.state[0,i] == 'O': # Element in top of column
                valid_moves.append(self.current_player+str(i+1)) # +1 to convert back to humanspeak
        
        return valid_moves

    def make_move(self, move: str):
        """
        Updates the game state to reflect the given move.
        Returns False if the move was not made and returns True if the move was made.
        """
        if move not in self.get_valid_moves():
            print("Invalid move, " + move + " attempted in position: ")
            self.show_game_state()
            return False
        for i in range(6):
            # The move[1]-1 is because we will let users assume columns 1 - 7 while we have 0 based indexing
            # The 5-i is so that we start at the bottom of the board
            if self.state[5-i,int(move[1])-1] == 'O':
                self.state[5-i,int(move[1])-1] = move[0] # Change the desired slot on board
                if self.current_player == 'R': # Update current player and state description
                    self.current_player = 'B'
                    self.description = 'Black to play.'
                else:
                    self.current_player = 'R'
                    self.description = 'Red to play.'
                return True # Return that it's a successful move
            if i == 5:
                print('End of column reached. Serious error.') # This should not be possible to reach
                return False 
    
    def show(self) -> None:
        print(self.description)
        print(self.state)