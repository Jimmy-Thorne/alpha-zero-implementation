class node():
    """A node carries all information in a tree, the weighted edge of the above up_edge
    i.e. the edge to its parent as well as current information."""

    def __init__(self, data:list, parent=None,is_root=False):
        self.data = data
        self.parent = parent
        self.N = 0
        self.p = 0
        self.Q = 0
        self.W = 0
        self.is_root = is_root

    def show_properties(self):
        print(f"N = {self.N}, p={self.p}, Q= {self.Q}, W={self.W}, moves = {self.data} ")



class wtree():

    def __init__(self):
        self.root = node([],None,True)
        self.children = {self.root:[]}
        self.node_list = [self.root]

    def add_node(self,node):
        '''Adds a node to the tree, if no node is passed it will assume the parent node is root'''
        #set blank nodes to have the root as a parent
        if node.parent==None:
            node.parent=self.root
        #Add new node to dictionary and node_list
        self.children[node] = []
        self.node_list.append(node)

        #keep the dictionary of children set
        if node.parent in self.children.keys():
            self.children[node.parent].append(node)
        else:
            self.children[node.parent] = [node]

        #print("Following Node Added")
        #node.show_properties()

a = wtree()
b = node(['B1'])
a.add_node(b)
