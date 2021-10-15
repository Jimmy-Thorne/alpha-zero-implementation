from connect_4_game import connect_4_game
from player import player
from random_player import random_player
from manual_player import manual_player
from connect_4_minimax_player import connect_4_minimax_player

num_black = 0
num_red = 0
num_draw = 0

for i in range(1):
    player1 = connect_4_minimax_player(3)
    player2 = random_player()

    game = connect_4_game(player1, player2)

    game.play()

    if game.board.state_description == 'Black wins.': num_black += 1
    elif game.board.state_description == 'Red wins.': num_red += 1
    else: num_draw += 1

    game.show_game_state()

print(num_black, num_red, num_draw)