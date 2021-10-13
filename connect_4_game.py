import numpy as np

from game import game

class connect_4_game(game):
    """
    Class that inherits the functionality of game, specifically implements for Connect 4
    """

    def __init__(self):
        """
        We initialize the game state to a 6 by 7 array of all 'O'
        """
        self.game_state = np.full((6,7),'O')
        self.state_description = 'Black to play'
        self.current_player = 'B'
        self.valid_moves = self.get_valid_moves()

    def check_for_finality(self):
        # First check for horizontal wins
        for i in range(6):
            for j in range(4):
                if self.game_state[i,j] == 'O': continue # Certainly isn't a win
                if self.game_state[i,j] == self.game_state[i,j+1] == self.game_state[i,j+2] == self.game_state[i,j+3]:
                    if self.game_state[i,j] == 'R':
                        self.state_description = 'Red wins.'
                    else:
                        self.state_description = 'Black wins.'
                    return True
        
        # Second check for vertical wins
        for i in range(3):
            for j in range(7):
                if self.game_state[i,j] == 'O': continue # Certainly isn't a win
                if self.game_state[i,j] == self.game_state[i+1,j] == self.game_state[i+2,j] == self.game_state[i+3,j]:
                    if self.game_state[i,j] == 'R':
                        self.state_description = 'Red wins.'
                    else:
                        self.state_description = 'Black wins'
                    return True
        
        # Third check for downwards diagonal wins
        for i in range(3):
            for j in range(4):
                if self.game_state[i,j] == 'O': continue # Certainly isn't a win
                if self.game_state[i,j] == self.game_state[i+1,j+1] == self.game_state[i+2,j+2] == self.game_state[i+3,j+3]:
                    if self.game_state[i,j] == 'R':
                        self.state_description = 'Red wins.'
                    else:
                        self.state_description = 'Black wins'
                    return True

        # Fourth check for upwards diagonal wins
        for i in range(3,6):
            for j in range(4):
                if self.game_state[i,j] == 'O': continue # Certainly isn't a win
                if self.game_state[i,j] == self.game_state[i-1,j+1] == self.game_state[i-2,j+2] == self.game_state[i-2,j+2]:
                    if self.game_state[i,j] == 'R':
                        self.state_description = 'Red wins.'
                    else:
                        self.state_description = 'Black wins'
                    return True

        # We have no connect 4's. Now check for a draw (no valid moves)
        if self.valid_moves == []:
            self.state_description = 'The game is a draw.'
            return True
        
        # Seems like the game can keep going
        return False

    def get_valid_moves(self):
        valid_moves = []

        for i in range(7):
            if self.game_state[0,i] == 'O':
                valid_moves.append(self.current_player+str(i+1)) # +1 to convert back to humanspeak
        
        return valid_moves

    def make_move(self, move):
        """
        Updates the game state to reflect the given move.
        Returns False if the move was not made and returns True if the move was made.
        """
        if move not in self.valid_moves:
            print('Invalid move attempted: ' + move + ', in position:')
            self.show_game_state()
            return False
        for i in range(6):
            # The move[1]-1 is because we will let users assume columns 1 - 7 while we have 0 based indexing
            # The 5-i is so that we start at the bottom of the board
            if self.game_state[5-i,int(move[1])-1] == 'O':
                self.game_state[5-i,int(move[1])-1] = move[0] # Change the desired slot on board
                if self.current_player == 'R': # Update current player and state description
                    self.current_player = 'B'
                    self.state_description = 'Black to play.'
                else:
                    self.current_player = 'R'
                    self.state_description = 'Red to play.'
                self.valid_moves = self.get_valid_moves() # Update valid moves
                return True # Return that it's a successful move
            if i == 5:
                print('End of column reached. Serious error.') # This should not be possible to reach
                return False
            
    def show_game_state(self):
        print(self.state_description)
        print(self.game_state)