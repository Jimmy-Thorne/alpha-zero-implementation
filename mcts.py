from board import board
from wtree import node
from wtree import wtree
from nn_temp import policy
from nn_temp import value
import copy

# For testing
from connect_4.connect_4_board import connect_4_board

class mcts():

    def __init__(self,current_board:board,depth:int):
        self.board = current_board
        self.depth = depth
        self.dummy_board = copy.deepcopy(current_board)  #TODO: Make this not by reference
        self.tree = wtree()
        self.valid_moves = self.board.get_valid_moves()

    def U_function(self,node):
        return node.p

    def find_next_node(self):
        chitlins = []
        for a in self.tree.children:
            if self.tree.children[a]==[]:
                chitlins.append(a)
        max = 0
        inx = 0
        for i in range(len(chitlins)):
            QplusU = chitlins[i].Q + self.U_function(chitlins[i])
            if QplusU > max:
                max = QplusU
                inx = i
        return chitlins[inx]

    def play_board(self,moves:list):
        for each in moves:
            self.dummy_board.make_move(each)


    def initial_population(self):
        if self.tree.children[self.tree.root]==[]:
            p = policy(self.dummy_board)
            for i in range(len(self.valid_moves)):
                k = node([self.valid_moves[i]])
                k.p = p[i]
                self.tree.add_node(k)

    def reset_dummy_board(self):
        self.dummy_board = copy.deepcopy(self.board)


    def back_prop(self,input_node,value):
        current_node = input_node
        while current_node.is_root == False:
            current_node.N = current_node.N + 1
            current_node.W = current_node.W + value
            current_node.Q = current_node.W / current_node.N
            current_node = current_node.parent


    def expand_node(self,current_node):
        p = policy(self.dummy_board)
        updated_valid_moves = self.dummy_board.get_valid_moves()
        new_data = current_node.data
        for i in range(len(updated_valid_moves)):
            new_data.append(updated_valid_moves[i])
            k = node(copy.copy(new_data),current_node)
            k.p = p[i]
            self.tree.add_node(k)
            new_data.pop(-1)


    def select_final_move(self,random=False):
        list_of_valid_nodes = self.tree.children[self.tree.root]
        max = list_of_valid_nodes[0].N
        indx = 0
        index = 0
        for each in list_of_valid_nodes:
            if max < each.N:
                max = each.N
                index = indx
            indx = indx + 1
        print(list_of_valid_nodes[index].data)
        return list_of_valid_nodes[index].data


    def run(self,random=False):
        self.reset_dummy_board()
        self.initial_population()
        for n in range(self.depth):
            k = self.find_next_node()
            self.play_board(k.data)
            self.expand_node(k)
            v = value(self.dummy_board)
            self.back_prop(k,v)
            self.reset_dummy_board()
        return self.select_final_move()



## TODO: Create a U function.  Check for vaild moves.  Add Random 'learning' process. Pass off a wtree.




brd = connect_4_board()
a = mcts(brd,100)
a.dummy_board.make_move('B1')
a.run()
