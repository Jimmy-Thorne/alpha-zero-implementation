from connect_4_game import connect_4_game
from random_player import random_player
from manual_player import manual_player
from minimax_player import minimax_player
import connect_4_value_functions

num_black = 0
num_red = 0
num_draw = 0

for i in range(10):
    player1 = minimax_player(connect_4_value_functions.naive, 3)
    player2 = minimax_player(connect_4_value_functions.naive, 3)

    game = connect_4_game(player1, player2)

    game.play(False)

    if game.board.description == 'Black wins.': num_black += 1
    elif game.board.description == 'Red wins.': num_red += 1
    else: num_draw += 1

print(num_black, num_red, num_draw)