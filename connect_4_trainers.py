from connect_4_game import connect_4_game
from connect_4_board import connect_4_board
from minimax_player import minimax_player
from player import player
from typing import Callable
import sys
import datetime
from nn import nn
from game import game
import tensorflow as tf
import copy
import numpy as np

class simple_trainer():
    """
    This trainer trains a neural network model against itself playing connect 4.
    Initialize the trainer by telling it where it should load/save the best model and how to construct the model.
    """
    def __init__(self, best_model_location: str, nn_constructor: Callable[[str,str],nn]) -> None:
        self.best_model_location = best_model_location
        self.nn_constructor = nn_constructor
        
    def test_best_nn(self, test_player: player, rounds: int = 10, clear_report: bool = False) -> None:
        """
        This plays this trainers best_nn against player rounds times.
        It then puts the report in simple_c4_trainer_reports.txt.
        It clears the report first if clear_report.
        """
        best_wins = 0
        test_wins = 0
        draws = 0

        best_nn = self.nn_constructor(self.best_model_location, name='Best connect 4 nn')
        nn_player = minimax_player(lambda x,y: best_nn(x,y), name='Best nn player')

        # Loop through the test rounds switching who is black and red each time
        for i in range(rounds):
            if i % 2 == 0:
                game = connect_4_game(nn_player, test_player)
            else:
                game = connect_4_game(test_player, nn_player)
        
            game.play(False)

            if game.winner == nn_player:
                best_wins += 1
            elif game.winner == test_player:
                test_wins += 1
            else:
                draws += 1
        
        # print a report of the test
        og = sys.stdout
            
        with open('connect_4_simple_trainer_report.txt', 'a') as f:
            if clear_report: f.truncate(0)
            sys.stdout = f
            print('--- Testing best connect 4 nn report at {0} ---'.format(datetime.datetime.now().strftime("%H:%M:%S")))
            print('\tBest nn wins: {0}'.format(best_wins))
            print('\t{1} wins: {0}'.format(test_wins, test_player.name))
            print('\tDraws: {0}'.format(draws))
            sys.stdout = og

    def train(self,
        batch_size = 1,
        epochs = 5,
        initial_load = True,
        num_test: int = 10,
        num_train: int = 100,
        only_load_best = False,
        rounds: int = 1,
        threshold: float = 0.55,
        verbose = True) -> None:
        # We will do rounds rounds of making a new training_nn, training it against itself,
        # and then comparing it to our best_nn. 
        for j in range(rounds):
            # If initial_load is False, we want to start with a random model on the first round.
            # If only_load_best, then we will always have the training net start from a blank slate.
            if not initial_load and j == 0:
                training_nn = self.nn_constructor(name = 'training_nn')
                best_nn = self.nn_constructor(name = 'best_nn')
            else:
                if only_load_best:
                    training_nn = self.nn_constructor(name = 'training_nn')
                    best_nn = self.nn_constructor(self.best_model_location, name = 'best_nn')
                else:
                    training_nn = self.nn_constructor(self.best_model_location, 'training_nn')
                    best_nn = self.nn_constructor(self.best_model_location, 'best_nn')

            # Set up the lists to hold training data
            x_train = []
            y_train = []

            # Now we will have the network play rounds against itself
            for i in range(num_train):

                # To begin a training loop, make a board
                current_board = connect_4_board()

                # Initialize the board list to only having the starting board
                board_list = [current_board]

                # Initialize the players
                black_player = minimax_player(lambda x,y: training_nn(x,y), name = 'Black player', key = 'B')
                red_player = minimax_player(lambda x,y: training_nn(x,y), name = 'Red player', key = 'R')

                # Set the current player to the black player for the following while loop
                current_player = black_player

                while True:
                    # First the current player will select a move for the current board.
                    move = current_player.choose_move(current_board)

                    # current_board.make_move(move) will always run.
                    # If it returns false we enter the if statement and declare the opposite player winner.
                    if not current_board.make_move(move):
                        print('Invalid Move Error')
                        break

                    # Check to see if the game is over.
                    if current_board.check_for_finality():
                        # Add it to the board list
                        board_list.append(copy.deepcopy(current_board))
                        break # Stop playing the game.

                    # Add it to the board list
                    board_list.append(copy.deepcopy(current_board))

                    # If a valid move was made and the game is not over, move to the next player.
                    if current_player == black_player:
                        current_player = red_player
                    else:
                        current_player = black_player

                # Encode the result in a number
                if current_board.description == 'Black wins.':
                    result = 1
                elif current_board.description == 'Red wins.':
                    result = -1
                else:
                    result = 0

                if verbose: print('Game {0}/{1} of round {2}/{3}.'.format(i+1, num_train, j+1, rounds), end='\r')

                # Now loop through the boards and prepare training data
                for board in board_list:
                    index = board_list.index(board)

                    # This value is 1 if we are on the last board in the list
                    # AKA the value function really should have known wtf was up.
                    # When we are on the first board, it dimishes the desired value significantly
                    # That way we aren't trying to convince the NN that it should have known
                    # that a certain player was going to win the game early on.
                    # I picked 30 just because that appears to be roughly the length of a quality game.
                    # Why exponential decay? Cause like... nature... man...
                    scaled_value = np.exp(-np.log(30) * ((len(board_list) - (index + 1)) / len(board_list)))

                    y_train.append(result * scaled_value)            
                    x_train.append(training_nn.board_to_array(board))

            # Actually fit the model
            training_nn.model.fit(np.array(x_train), np.array(y_train), batch_size = batch_size, epochs = epochs, shuffle=False, verbose = 0)

            # Now begin the test rounds to decide if the training net can beat the current best threshold% of the time
            trained_wins = 0
            best_wins = 0

            for i in range(num_test):
                if i % 2 == 0:
                    player1 = minimax_player(lambda x,y: training_nn(x,y), 1, name = 'train')
                    player2 = minimax_player(lambda x,y: best_nn(x,y), 1, name = 'best')
                else:
                    player1 = minimax_player(lambda x,y: best_nn(x,y), 1, name = 'best')
                    player2 = minimax_player(lambda x,y: training_nn(x,y), 1, name = 'train')      

                game = connect_4_game(player1, player2)

                game.play(False)

                if game.board.description == 'Black wins.':
                    if player1.name == 'train':
                        trained_wins += 1
                    else:
                        best_wins += 1
                elif game.board.description == 'Red wins.':
                    if player1.name == 'train':
                        best_wins += 1
                    else:
                        trained_wins += 1

            # Save whoever is best as best
            if (trained_wins / num_test) >= threshold:
                training_nn.save(self.best_model_location)
                if verbose: print('\nNew best network!')
            else:
                best_nn.save(self.best_model_location)
                if verbose: print('\nThe current best remains king!')