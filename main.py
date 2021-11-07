from random_player import random_player
from manual_player import manual_player
from minimax_player import minimax_player
import connect_4_value_functions
import connect_4_nn_constructors as c4c
import connect_4_trainers as c4t

import numpy as np
import copy
import sys
import datetime

trainer = c4t.simple_trainer('models\\connect_4\\best_c4_simple_conv', c4c.simple_conv_connect_4_nn)

trainer.train(batch_size = 20, initial_load=False, rounds = 5)

test_player = minimax_player(connect_4_value_functions.naive, 2)

trainer.test_best_nn(test_player, rounds=100)