import numpy as np
import tensorflow as tf
import copy
import sys
import datetime

from connect_4_board import connect_4_board
from minimax_player import minimax_player
from player import player
from nn import nn

def convert_board_to_tensor(board: connect_4_board):
    """
    Function creates a (6,7,2) tensor from the board.
    The first matrix is keeping track of pieces, the second whose turn it is.
    """

    # We might as well initialize it with the knowledge of what the second matrix should be
    if board.current_player == 'B':
        lst = np.ones((6,7,2))
    else:
        lst = -1 * np.ones((6,7,2))

    # Now fix the first matrix
    for i in range(6):
        for j in range(7):
            if board.state[i,j] == 'B': lst[i,j,0] = 1
            elif board.state[i,j] == 'R': lst[i,j,0] = -1
            else: lst[i,j,0] = 0

    # return is as a numpy array as this is what tensorflow wants
    return np.array(lst)

class connect_4_nn(nn):
    """
    The class wrapping a neural network to evaluate a connect_4_board
    """

    def __init__(self, checkpoint_name: str = None, name: str = 'connect_4_nn', expect_partial: bool = False) -> None:
        """ Wraps a nn to evaluate a connect_4_board. checkpoint_name is where to save/load this nets weights.
        If expect_partial, then it will not load the training variables from the checkpoint.
        This is used when you only want to make the network for inference rather than training."""
        inputs = tf.keras.Input(shape = (6,7,2))

        # For this first layer we have the 4 possible connect 4s we'd like to detect.
        # This is why we are using (4,4) kernel size
        # Not sure what activations to use here...
        outputs = tf.keras.layers.Conv2D(4, (4,4), activation = 'linear',
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            bias_initializer=tf.keras.initializers.RandomNormal(),
            bias_regularizer=tf.keras.regularizers.l1(0.1))(inputs)
        outputs = tf.keras.layers.Flatten()(outputs)
        outputs = tf.keras.layers.Dense(1, activation = 'tanh',
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            bias_initializer='zeros',
            bias_regularizer=tf.keras.regularizers.l2(0.1))(outputs)

        self.model = tf.keras.Model(inputs = inputs, outputs = outputs)

        loss_fn = tf.keras.losses.MeanSquaredError()

        self.model.compile(optimizer='adam', loss=loss_fn)

        if checkpoint_name is not None:
            try:
                if expect_partial:
                    self.model.load_weights('checkpoints\\{0}'.format(checkpoint_name)).expect_partial()
                else:
                    self.model.load_weights('checkpoints\\{0}'.format(checkpoint_name))
            except:
                print('Given best_checkpoint doesn\'t exist. Proceed with random weights?')
                x = input('y or n?')
                if x == 'n':
                    exit()

        self.checkpoint_name = checkpoint_name
        self.name = name

    def __call__(self, board: connect_4_board, player: player):
        board_value = self.model(np.array([convert_board_to_tensor(board)]))

        if player.key == 'B':
            return board_value
        else:
            return -1 * board_value

    def __str__(self) -> str:
        return '{0} with checkpoint {1}'.format(self.name, self.checkpoint_name)

    def load(self, alternative_checkpoint_name = None) -> None:
        if self.checkpoint_name is None and alternative_checkpoint_name is None:
            print('No specified checkpoint name! Cannot load {0}'.format(str(self)))
        elif alternative_checkpoint_name is not None:
            self.model.load_weights('checkpoints\\{0}'.format(alternative_checkpoint_name))
        else:
            self.model.load_weights('checkpoints\\{0}'.format(self.checkpoint_name))

    def save(self, alternative_checkpoint_name = None) -> None:
        if self.checkpoint_name is None and alternative_checkpoint_name is None:
            print('No specified checkpoint name! Cannot save {0}'.format(str(self)))
        elif alternative_checkpoint_name is not None:
            self.model.save_weights('checkpoints\\{0}'.format(alternative_checkpoint_name))
        else:
            self.model.save_weights('checkpoints\\{0}'.format(self.checkpoint_name))

    def train(self, rounds = 1, batch_size = 1, epochs = 5):
        # Set up the lists to hold training data
        x_train = []
        y_train = []

        # Now we will have the network play rounds against itself
        for i in range(rounds):

            # To begin a training loop, make a board
            current_board = connect_4_board()

            # Initialize the board list to only having the starting board
            board_list = [current_board]

            # Initialize the players
            black_player = minimax_player(lambda x,y: self(x,y), name = 'Black player', key = 'B')
            red_player = minimax_player(lambda x,y: self(x,y), name = 'Red player', key = 'R')

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
                x_train.append(convert_board_to_tensor(board))

        # Actually fit the model
        self.model.fit(np.array(x_train), np.array(y_train), batch_size = batch_size, epochs = epochs, shuffle=False)