import enum
import numpy as np
import random
from collections import Counter
from gettingStarted import impurity, bestsplit
'''
### The structure of the tree follows a set of nodes, each with a comparator assigned to it.
### To classify, each node operates a value of a certain attribute (column), against a constant
### or another attribute, and then diverges into its left child (if the result is True) or
### into its right child (if the result is False)
###
### The structure of the tree allows for more comparators than just left/right splitters,
### as well as comparisons between attributes, but for the former all but one are unused,
### and for the latter it is not used, since the assignment solicits that bestsplit(x,y)
### should only look for left/right splits on numeric constants
###
### The classes comprising the tree are ClassificationTreeDM (tree), CTreeDMNode (node)
### and CTreeDMLeaf (leaf)
'''
'''
### Enum OpType
###
### These are the six operators a tree can use for each node. Each operator
### assigned a function to compare two values
'''


class OpType(enum.Enum):
    EQUALS = 1
    LESSTHAN = 2
    LESSTHANEQUALS = 3
    MORETHAN = 4
    MORETHANEQUALS = 5
    NOTEQUALS = 6

    comparators = {
        EQUALS: lambda x, y: True if x == y else False,
        LESSTHAN: lambda x, y: True if x < y else False,
        LESSTHANEQUALS: lambda x, y: True if x <= y else False,
        MORETHAN: lambda x, y: True if x > y else False,
        MORETHANEQUALS: lambda x, y: True if x >= y else False,
        NOTEQUALS: lambda x, y: True if x != y else False
    }


'''
### ClassificationTreeDM: Tree class
###
### Binary classification tree, with nodes that propagate either
### left or right based on comparisons to data fed
###
###
'''


class ClassificationTreeDM(object):
    '''
    ### Constructor
    ###
    ### Trains the tree with the data passed by argument
    ###
    ### Arguments:
    ###     values: 2-D Numpy matrix of individuals as rows and attributes as columns
    ###     labels: 1-D Array of classes for each row of values
    ###     nmin: Minimum number of individuals represented by a splitting node
    ###     minleaf: Minimum number of individuals contained on a leaf to exist
    ###     nfeat: Number of features to consider
    '''
    def __init__(self, values, labels, nmin, minleaf, nfeat):
        self.root = self.produceNextNode(values, labels, nmin, minleaf, nfeat)

    def produceNextNode(self,
                        values,
                        labels,
                        nmin,
                        minleaf,
                        nfeat,
                        parent=None):
        ## If all elements are of the same class, no need to split
        if len(np.unique(labels)) < 2:
            return CTreeDMLeaf(labels)

        ## Check minimum size for this to be a node
        ## else, become a leaf
        if values.shape[0] < nmin:
            return CTreeDMLeaf(labels)

        nextsplit = []
        ## Perform bagging if necessary
        if nfeat < values.shape[1] and nfeat != 0:
            attrs = random.sample(range(values.shape[1]), nfeat)
        else:
            attrs = range(values.shape[1])

        for i in attrs:
            nextattr = values[:, i]
            bsplit = bestsplit(nextattr, labels)
            nextsplit.append((bsplit["value"], bsplit["combined-gini"], i))

        nextsplit = sorted(nextsplit, key=lambda x: x[1])
        for i in range(len(nextsplit)):
            cutvalue, gini, column = nextsplit[i]

            ## Check sizes, if they are too small to even be
            ## a leaf, choose next best split
            nleft = values[values[:, column] <= cutvalue].shape[0]
            nright = values[values[:, column] > cutvalue].shape[0]
            if nleft >= minleaf and nright >= minleaf:
                ret = CTreeDMNode(parent,
                                  OpType.comparators.value[OpType.LESSTHANEQUALS.value],
                                  column,
                                  hasConstant=True,
                                  value=cutvalue)
                ret.setLeft(
                    self.produceNextNode(values[values[:, column] <= cutvalue],
                                         labels[values[:, column] <= cutvalue],
                                         nmin,
                                         minleaf,
                                         nfeat,
                                         parent=ret))
                ret.setRight(
                    self.produceNextNode(values[values[:, column] > cutvalue],
                                         labels[values[:, column] > cutvalue],
                                         nmin,
                                         minleaf,
                                         nfeat,
                                         parent=ret))
                return ret
        return CTreeDMLeaf(labels)

    def predict(self, row):
        curnode = self.root
        while not curnode.isLeaf:
            if curnode.hasConstant:
                ret = curnode.operator(row[curnode.columnIndex],
                                       curnode.compvalue)
            else:
                ret = curnode.operator(row[curnode.columnIndex],
                                       row[curnode.secondIndex])
            curnode = curnode.left if ret else curnode.right
        ret = curnode.getMajorityClass()
        return ret

    def __str__(self):
        return "Printing Tree:\nRoot " + self.root.print()


'''
### CTreeDMNode: Non-leaf node class
###
### Tree node class that stores the attribute it operates on,
### the comparator that node applies, the value it compares the attribute to,
### and left and right nodes for the node's children
###
### hasConstant and secondIndex are used to compare between attributes
### instead to numeric constants (currently unused)
'''


class CTreeDMNode(object):

    isLeaf = False

    def __init__(self,
                 parent,
                 operator,
                 columnIndex,
                 hasConstant=True,
                 value=None,
                 secondIndex=None):
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

    def print(self, tabs=0):
        return "".join(["\t" for i in range(tabs)]) + "Node. Column: " + str(
            self.columnIndex) + "\t- Value on split: " + str(
                self.compvalue) + "\n" + self.left.print(
                    tabs + 1) + self.right.print(tabs + 1)


'''
### CTreeDMLeaf: Leaf class
###
### Stores an array with all the classes for the individuals
### the leaf represents
'''


class CTreeDMLeaf(object):

    isLeaf = True

    def __init__(self, elements):
        self.cnt = Counter(elements)

    def getMajorityClass(self):
        return self.cnt.most_common(1)[0][0]

    def print(self, tabs=0):
        return "".join(["\t" for i in range(tabs)]) + "Leaf. Elements: " + str(
            self.cnt)
