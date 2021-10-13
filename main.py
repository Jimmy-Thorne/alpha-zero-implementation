from connect_4_game import connect_4_game

new_game = connect_4_game()

new_game.make_move('B1')
new_game.make_move('R2')
new_game.make_move('B2')
new_game.make_move('R3')
new_game.make_move('B5')
new_game.make_move('R3')
new_game.make_move('B3')
new_game.make_move('R4')
new_game.make_move('B4')
new_game.make_move('R4')
new_game.make_move('B4')

if new_game.check_for_finality():
    new_game.show_game_state()
else:
    print('huh')
    new_game.show_game_state()