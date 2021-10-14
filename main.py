from connect_4_game import connect_4_game
from player import player
from random_player import random_player
from manual_player import manual_player
from connect_4_minimax_player import connect_4_minimax_player

num_black = 0
num_red = 0
num_draw = 0

for i in range(10):

    player1 = random_player()
    player2 = connect_4_minimax_player(4)

    game = connect_4_game(player1, player2)

    while not game.check_for_finality():    
        player1.make_move(game)
        if game.check_for_finality():
            break
        player2.make_move(game)

    if game.state_description ==  'Red wins.': num_red += 1
    elif game.state_description == 'Black wins.': num_black += 1
    else: num_draw += 1

print(num_black)
print(num_red)
print(num_draw)