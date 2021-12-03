from random_player import random_player
from manual_player import manual_player
from minimax_player import minimax_player
from connect_4.connect_4_game import connect_4_game
import connect_4.connect_4_value_functions as c4v
import connect_4.connect_4_nn_constructors as c4c
import connect_4.connect_4_trainers as c4t
from nn import nn

import numpy as np
import copy
import sys
import datetime

trainer = c4t.simple_trainer('connect_4\\models\\10000_d4_trained_simple_conv', c4c.simple_conv_connect_4_nn)

"""player1 = minimax_player(c4v.naive, depth=2)
player2 = minimax_player(c4v.naive, depth=2)

trainer.generate_games(
    'connect_4\\games\\minimax_d2_10000_games.pkl',
    player1=player1,
    player2=player2,
    num_games=10000)"""
trainer.train_from_file(
    'connect_4\\games\\minimax_d4_10000_scaled_games.pkl',
    initial_load=False,
    early_stopping=True,
    tensorboard=False,
    epochs=4,
    num_test=10,
    batch_size=64)

best_nn = c4c.simple_conv_connect_4_nn('connect_4\\models\\10000_d4_trained_simple_conv')

player1 = minimax_player(c4v.naive, depth=2, name='Naive player')
player2 = minimax_player(best_nn, depth=2, name='nn player')

"""trainer.train(
    initial_load=True,
    load_history_from='connect_4\\c4_history.pkl',
    batch_size = 32,
    num_self_play=2,
    test_every=4,
    rounds=1,
    tensorboard=True
)"""

"""trainer.test_best_nn(player1, rounds=100)"""

c4game = connect_4_game(player1, player2)
c4game.play_many(rounds=5)