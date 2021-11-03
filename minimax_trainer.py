import numpy as np
import tensorflow as tf
import copy
import sys
import datetime

from connect_4_board import connect_4_board
import connect_4_value_functions
from minimax_player import minimax_player
from random_player import random_player
from player import player

def convert_board_to_tensor(board: connect_4_board):
    """
    Function turns 'B' into 1, 'R' into -1, and 'O' into 0.
    The second array is all 1 if black is the current player and all -1 if red is
    """
    lst = []

    for i in range(6):
        for j in range(7):
            if board.state[i,j] == 'B': lst.append(1)
            elif board.state[i,j] == 'R': lst.append(-1)
            else: lst.append(0)

    if board.current_player == 'B':
        lst.append(1)
    else:
        lst.append(-1)

    return np.array(lst)


class minimax_trainer():
    """
    The class designed to train the neural network for minimax players
    """

    def __init__(self) -> None:
        # For this model we will flatten the board into a 42 vector,
        # then add an additional number indicating whose turn it is
        inputs = tf.keras.Input(shape = (43,))
        outputs = tf.keras.layers.Dense(43, activation = 'linear',
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
            bias_initializer=tf.keras.initializers.RandomNormal(),
            bias_regularizer=tf.keras.regularizers.l1(0.01))(inputs)
        outputs = tf.keras.layers.Dense(24+21+24, activation = 'relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            bias_initializer=tf.keras.initializers.RandomNormal(),
            bias_regularizer=tf.keras.regularizers.l1(0.001))(outputs)
        outputs = tf.keras.layers.Dense(1, activation = 'tanh',
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            bias_initializer='zeros')(outputs)

        self.model = tf.keras.Model(inputs = inputs, outputs = outputs)

        loss_fn = tf.keras.losses.MeanSquaredError()

        self.model.compile(optimizer='adam', loss=loss_fn)

    def value_function(self, board: connect_4_board, player: player):
        board_value = self.model.predict(np.array([convert_board_to_tensor(board)]))

        if player.key == 'B':
            return board_value
        else:
            return -1 * board_value

    def train(self, debug = False, rounds = 1):
        # Set variables to keep track of wins throughout training.
        network_wins = 0
        naive_wins = 0
        draws = 0

        # Set up the lists to hold training data
        x_train = []
        y_train = []

        # A bool for our loop which keeps track of whether the network player
        # or the naive player are going first.
        net_first = True
        for i in range(rounds):

            # To begin a training loop, make a board
            current_board = connect_4_board()

            # Initialize the board list to only having the starting board
            board_list = [current_board]

            # Pick who the black and red players are depending upon whether the network is black (goes first)
            if net_first:
                black_player = minimax_player(name = 'Network player', value_function = self.value_function)
                black_player.key = 'B'

                red_player = minimax_player(name = 'Naive player', value_function = connect_4_value_functions.naive, depth = 2)
                red_player.key = 'R'

                net_first = False
            else:
                black_player = minimax_player(name = 'Naive player', value_function = connect_4_value_functions.naive, depth = 2)
                black_player.key = 'B'

                red_player = minimax_player(name = 'Network player', value_function = self.value_function)
                red_player.key = 'R'

                net_first = True

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
                if black_player.name == 'Network player':
                    network_wins += 1
                else:
                    naive_wins += 1
            elif current_board.description == 'Red wins.':
                result = -1
                if black_player.name == 'Network player':
                    naive_wins += 1
                else:
                    network_wins += 1
            else:
                result = 0
                draws += 1



            for board in board_list:
                y_train.append(result)            
                x_train.append(convert_board_to_tensor(board))

            if debug:
                games = naive_wins + network_wins + draws
                print('Round {3}/{4} - Nai: {0}, Net: {1}, Draw: {2}'.format(np.round(naive_wins / games, 2),
                                                            np.round(network_wins / games, 2),
                                                            np.round(draws / games, 2),
                                                            games, rounds))
        # Actually fit the model
        self.model.fit(np.array(x_train), np.array(y_train), batch_size = 2, epochs = 50, shuffle=False)
        
        # For training purposes append the training report into results
        og = sys.stdout

        total = naive_wins + network_wins + draws

        with open('training_results.txt', 'w') as f:
            sys.stdout = f
            print('--- Training Win Report at {0} ---'.format(datetime.datetime.now().strftime("%H:%M:%S")))
            print('\tNaive wins: {0} -- {1}%'.format(naive_wins, np.round(naive_wins / total, 2)))
            print('\tNetwork wins: {0} -- {1}%'.format(network_wins, np.round(network_wins / total,2)))
            print('\tDraws: {0} -- {1}%'.format(draws, np.round(draws / total, 2)))
        
        sys.stdout = og