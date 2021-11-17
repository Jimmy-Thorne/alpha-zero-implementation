from random_player import random_player
from manual_player import manual_player
from minimax_player import minimax_player
import connect_4.connect_4_value_functions as c4v
import connect_4.connect_4_nn_constructors as c4c
import connect_4.connect_4_trainers as c4t
from nn import nn

import numpy as np
import copy
import sys
import datetime

trainer = c4t.simple_trainer('connect_4\\models\\best_c4_simple_conv', c4c.simple_conv_connect_4_nn)

player1 = minimax_player(c4v.naive, depth=2)
player2 = minimax_player(c4v.naive, depth=2)

#trainer.generate_games('connect_4\\minimax_d2_1000_games.pkl', player1=player1,player2=player2, num_games=1000)
"""trainer.train_from_file(
    'connect_4\\minimax_d2_1000_games.pkl',
    initial_load=False,
    tensorboard=True,
    epochs=16,
    num_test=100,
    batch_size=32)"""

trainer.train(
    initial_load=True,
    load_history_from='connect_4\\c4_history.pkl',
    num_self_play=2,
    test_every=16,
    rounds=2,
    tensorboard=False
)

trainer.test_best_nn(player1, rounds=10)