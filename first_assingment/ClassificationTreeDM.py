import enum
from collections import Counter

'''
### The structure of the tree follows a set of nodes, each with a comparator assigned to it.
### To classify, each node operates a value of a certain attribute (column), against a constant
### or another attribute, and then diverges into its left child (if the result is True) or
### into its right child (if the result is False)
'''

class OpType(enum.Enum):
    EQUALS = 1
    LESSTHAN = 2
    LESSTHANEQUALS = 3
    MORETHAN = 4
    MORETHANEQUALS = 5
    NOTEQUALS = 6
    
    comparators = {EQUALS:lambda x,y: True if x == y else False,
                    LESSTHAN:lambda x,y: True if x < y else False,
                    LESSTHANEQUALS:lambda x,y: True if x <= y else False,
                    MORETHAN:lambda x,y: True if x > y else False,
                    MORETHANEQUALS:lambda x,y: True if x >= y else False,
                    NOTEQUALS:lambda x,y: True if x != y else False}



class ClassificationTreeDM(object): 

    ## TODO
    def __init__(self):
        pass

    def predict(self, row):
        curnode = self.root
        while not curnode.isLeaf:
            if curnode.hasConstant:
                ret = curnode.operator(row[curnode.columnIndex], curnode.compvalue)
            else:
                ret = curnode.operator(row[curnode.columnIndex], row[curnode.secondIndex])
            curnode = curnode.left if ret else curnode.right

        return curnode.getMajorityClass()
    

## Mostly TODO
class CTreeDMNode(object):

    def __init__(self, parent, operator, columnIndex, hasConstant, value=None, secondIndex=None):
        self.parent = parent
        self.operator = operator
        self.compvalue = value
        self.hasConstant = hasConstant
        self.columnIndex = columnIndex
        self.secondIndex = secondIndex

    def setLeft(self, node):
        self.left = node

    def setRight(self, node):
        self.right = node

'''
### CTreeDMLeaf: Leaf class
###
### Stores an array with all the individuals for each
### class it represents
'''
class CTreeDMLeaf(object):

    isLeaf = True

    def __init__(self, elements):
        self.elements = elements
        self.cnt = Counter(self.elements)

    def getMajorityClass(self):
        return self.cnt.most_common(1)[0][0]
