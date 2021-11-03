from operator import is_
from connect_4_board import connect_4_board
from connect_4_game import connect_4_game
from random_player import random_player
from manual_player import manual_player
from minimax_player import minimax_player
import connect_4_value_functions
from minimax_trainer import minimax_trainer
import numpy as np
import copy

def trainloop(train_loops = 100, test_loops = 10, load_weights = False):
    num_trained = 0
    num_naive = 0
    num_draw = 0

    trainer = minimax_trainer()

    if load_weights:
        weights = []
        weights2 = trainer.model.get_weights()

        for i in range(6):
            tmp = np.genfromtxt('weights{0}.csv'.format(i))
            tmp = np.reshape(tmp, weights2[i].shape)
            weights.append(tmp)
    
        trainer.model.set_weights(weights)

    minimax_player_key = 'B'

    trainer.train(debug = True, rounds = train_loops)

    weights = trainer.model.get_weights()

    for i in range(len(weights)):
        np.savetxt('weights{0}.csv'.format(i), weights[i])

    for i in range(test_loops):
        if i % 2 == 0:
            player1 = minimax_player(trainer.value_function, 2, name = 'train')
            player2 = minimax_player(connect_4_value_functions.naive, 2, name = 'naive')
        else:
            player1 = minimax_player(connect_4_value_functions.naive, 2, name = 'naive')
            player2 = minimax_player(trainer.value_function, 2, name = 'train')      

        game = connect_4_game(player1, player2)

        game.play(False)

        if game.board.description == 'Black wins.':
            if player1.name == 'train':
                num_trained += 1
            else:
                num_naive += 1
        elif game.board.description == 'Red wins.':
            if player1.name == 'train':
                num_naive += 1
            else:
                num_trained += 1
        else:
            num_draw +=1

    print('--- Testing Win Report ---')
    print('\tNaive wins: {0}'.format(num_naive))
    print('\tNetwork wins: {0}'.format(num_trained))
    print('\tDraws: {0}'.format(num_draw))

def debug_net():
    trainer = minimax_trainer()

    weights = []
    weights2 = trainer.model.get_weights()

    for i in range(6):
        tmp = np.genfromtxt('weights{0}.csv'.format(i))
        tmp = np.reshape(tmp, weights2[i].shape)
        weights.append(tmp)

    trainer.model.set_weights(weights)

    player1 = manual_player(key='B')
    player2 = random_player(key='R')

    is_game = False
    board = connect_4_board()
    list = []

    while not is_game:
        copy_board_state = copy.deepcopy(board.state)
        list.append((copy_board_state, trainer.value_function(board, player1)))

        board.make_move(player1.choose_move(board))

        copy_board_state = copy.deepcopy(board.state)
        list.append((copy_board_state, trainer.value_function(board, player2)))

        is_game = board.check_for_finality()

        if is_game: break

        board.make_move(player2.choose_move(board))

        is_game = board.check_for_finality()

    print(list)

for i in range(5):
    trainloop(100, 5, load_weights = True)