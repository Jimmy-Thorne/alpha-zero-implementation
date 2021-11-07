import numpy as np
import tensorflow as tf
import copy
import sys
import datetime
from typing import Callable
from typing import Tuple

from connect_4_board import connect_4_board
from minimax_player import minimax_player
from player import player
from nn import nn
from board import board
from nn import nn

def simple_conv_connect_4_nn(model_location: str = None, name: str = 'simple_conv_connect_4_nn') -> nn:
    """
    Constructs a simple convolutional neural network for connect 4.
    """
    inputs = tf.keras.Input(shape = (6,7,2))
    outputs = tf.keras.layers.Conv2D(4, (4,4), activation = 'linear',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer=tf.keras.initializers.RandomNormal(),
        bias_regularizer=tf.keras.regularizers.l1(0.1))(inputs)
    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.Dense(1, activation = 'tanh',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer='zeros',
        bias_regularizer=tf.keras.regularizers.l2(0.1))(outputs)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer='adam', loss=loss_fn)

    if model_location is not None:
        try:
            model = tf.keras.models.load_model(model_location)
        except:
            print('Failed to load model from {0}. Continue with random weights for {1}?'.format(model_location, name))
    
    # Create the board conversion function
    def convert_board_to_tensor(board: connect_4_board) -> np.array:
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

    return nn(model, convert_board_to_tensor, name)