from operator import is_
from connect_4_board import connect_4_board
from connect_4_game import connect_4_game
from random_player import random_player
from manual_player import manual_player
from minimax_player import minimax_player
import connect_4_value_functions
from simple_trainer import simple_c4_trainer

import numpy as np
import copy
import sys
import datetime

trainer = simple_c4_trainer('best_c4_checkpoint')

trainer.train(batch_size=20, rounds=10)

test_player = minimax_player(connect_4_value_functions.naive, 2)

trainer.test_best_nn(test_player, rounds=100)