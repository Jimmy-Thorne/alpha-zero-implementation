#This is a temporary file for the neural network
from random import random

<<<<<<< HEAD
from connect_4.connect_4_board import connect_4_board
=======
from connect_4_board import connect_4_board
>>>>>>> nsalmon3-main
def policy(current_board):

    Valid_Moves = current_board.get_valid_moves()
    exceptions = []
    for i in range(7):
        if current_board.state[0,i] != 'O': # Element in top of column
            exceptions.append(i) # +1 to convert back to humanspeak

    n = len(Valid_Moves)
    probs = []
    for i in range(n):
        k = random()
        probs.append(k)
    tot = sum(probs)
    for i in range(n):
        probs[i] = probs[i]/tot

    policy = []
    j = 0
    for i in range(7):
        if i in exceptions:
            policy.append(0)
        else:
            policy.append(probs[0])
            probs.pop(0)

    return policy

def value(current_board):
    return random()
