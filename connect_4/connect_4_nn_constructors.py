import numpy as np
import tensorflow as tf
import scipy.signal

from connect_4.connect_4_board import connect_4_board
from minimax_player import minimax_player
from player import player
from nn import nn

def convolutional_layer(inputs):
    outputs = tf.keras.layers.Conv2D(64,(3,3), padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        bias_initializer=tf.keras.initializers.RandomNormal())(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    return tf.keras.activations.relu(outputs)

def residual_layer(inputs):
    outputs = tf.keras.layers.Conv2D(64,(3,3),padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        bias_initializer=tf.keras.initializers.RandomNormal())(inputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.activations.relu(outputs)

    outputs = tf.keras.layers.Conv2D(64,(3,3),padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        bias_initializer=tf.keras.initializers.RandomNormal())(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Add()([inputs,outputs])
    return tf.keras.activations.relu(outputs)

def simple_conv_connect_4_nn(model_location: str = None, name: str = 'simple_conv_connect_4_nn') -> nn:
    """
    Constructs a simple convolutional neural network for connect 4.
    """
    inputs = tf.keras.Input(shape = (6,7,2))
    outputs = convolutional_layer(inputs)
    outputs = residual_layer(outputs)
    outputs = tf.keras.layers.Conv2D(1,(1,1),padding='same')(outputs)
    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dense(1, activation = 'tanh',
        kernel_initializer=tf.keras.initializers.RandomNormal())(outputs)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

    model.compile(optimizer=optimizer, loss=loss_fn)

    if model_location is not None:
        try:
            model = tf.keras.models.load_model(model_location)
        except:
            print('Failed to load model from {0}. Continuing with random weights.'.format(model_location, name))
    
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

def pre_conv_connect_4_nn(model_location: str = None, name: str = 'pre_conv_connect_4_nn') -> nn:
    """
    Constructs a simple neural network for connect 4, where the inputs are pre-built convolutions.
    """
    inputs = tf.keras.Input(shape = (6,7,5))
    outputs = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(6*7*5, activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        bias_initializer=tf.keras.initializers.RandomNormal())(outputs)
    outputs = tf.keras.layers.Dense(16, activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        bias_initializer=tf.keras.initializers.RandomNormal())(outputs)
    outputs = tf.keras.layers.Dense(8, activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        bias_initializer=tf.keras.initializers.RandomNormal())(outputs)
    outputs = tf.keras.layers.Dense(1, activation = 'tanh',
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        bias_initializer=tf.keras.initializers.RandomNormal())(outputs)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    loss_fn = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss=loss_fn)

    if model_location is not None:
        try:
            model = tf.keras.models.load_model(model_location)
        except:
            print('Failed to load model from {0}. Continuing with random weights.'.format(model_location, name))
    
    # Create the board conversion function
    def convert_board_to_tensor(board: connect_4_board) -> np.array:
        # We might as well initialize it with the knowledge of what the last matrix should be
        if board.current_player == 'B':
            lst = np.ones((6,7))
        else:
            lst = -1 * np.ones((6,7))

        # We will hold the board in here for now
        lst2 = np.ones((6,7))

        # Now fix the first matrix
        for i in range(6):
            for j in range(7):
                if board.state[i,j] == 'B': lst2[i,j] = 1
                elif board.state[i,j] == 'R': lst2[i,j] = -1
                else: lst2[i,j] = 0

        # We will prep the input first by convolving with 4 filters
        filter1 = [[0,0,0],[1,1,1],[0,0,0]]
        filter2 = [[0,1,0],[0,1,0],[0,1,0]]
        filter3 = [[0,0,1],[0,1,0],[1,0,0]]
        filter4 = [[1,0,0],[0,1,0],[0,0,1]]

        conv1 = scipy.signal.convolve2d(lst2,filter1,mode='same')
        conv2 = scipy.signal.convolve2d(lst2,filter2,mode='same')
        conv3 = scipy.signal.convolve2d(lst2,filter3,mode='same')
        conv4 = scipy.signal.convolve2d(lst2,filter4,mode='same')
        
        lst = [conv1,conv2,conv3,conv4,lst]

        lst2 = np.ones((6,7,5))

        # Awkward reshaping...
        for i in range(6):
            for j in range(7):
                for k in range(5):
                    lst2[i,j,k] = lst[k][i,j]

        # return is as a numpy array as this is what tensorflow wants
        return np.array(lst2)

    return nn(model, convert_board_to_tensor, name)