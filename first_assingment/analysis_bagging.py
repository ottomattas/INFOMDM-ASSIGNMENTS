import numpy as np
import math
import enum
import random
from tqdm import tqdm
from collections import Counter

NUMEXECUTIONS = 100

'''
### tree_grow()
### Grow a classification tree.
### 
### Arguments:
###     x: 2-dimensional array of attribute values
###     y: 1-dimensional array of binary class labels
###     nmin: Min. number of observations required per node
###     minleaf: Min. number of observations to be a leaf node
###     nfeat: Number of features to consider on every split
###
### Returns:
###     A ClassificationTreeDM object with the generated tree
'''
def tree_grow(x, y, nmin, minleaf, nfeat=None, classnames=None):
    if nfeat is None:
        nfeat = len(x)
    return ClassificationTreeDM(x, y, nmin, minleaf, nfeat, classnames=classnames)



'''
### tree_pred()
### Make predicitions using the tree.
###
### Arguments:
###     x: 2-dimensional array of attribute values for predictions
###     tr: Tree object with a predict(row) method
###
### Returns:
###     List of class predictions for each element as predicted by the tree
'''
def tree_pred(x, tr):
    predictions = []

    for row in x:
        pred = tr.predict(row)
        predictions.append(pred)

    return predictions


'''
### tree_grow_b()
### Grow classification trees for bagging.
###
### Arguments:
###     x: 2-dimensional array of attribute values
###     y: 1-dimensional array of binary class labels
###     nmin: Min. number of observations required per node
###     minleaf: Min. number of observations to be a leaf node
###     nfeat: Number of features to consider on every split
###     m: Number of bootstrap samples to generate
###
### Returns:
###     A list of m ClassificationTreeDM objects each trained with a
###     bootstrap sample of the data
'''
def tree_grow_b(x, y, nmin, minleaf, nfeat=None, m=1, classnames=None):
    if nfeat is None:
        nfeat = x.shape[0]
    ret = []
    for i in tqdm(range(m)):
        indexes = np.random.choice(range(x.shape[0]), size=x.shape[0])
        ret.append(
            tree_grow(x[indexes, :], y[indexes], nmin, minleaf, nfeat, classnames=classnames))
    return ret


'''
### tree_pred_b()
###
### Arguments:
###     x: 2-dimensional array of attribute values for predictions
###     trees: List of tree objects with a predict(row) method
###
### Returns:
###     List of class predictions for each element decided
###     by majority vote between all the trees
'''
def tree_pred_b(x, trees):
    nelements = x.shape[0]
    totalpreds = None

    ## Obtain each prediction and put them into a
    ## numpy matrix, each row is a tree's prediction list
    for tree in trees:
        treepred = np.array(tree_pred(x, tree))
        if totalpreds is None:
            totalpreds = treepred
        else:
            totalpreds = np.vstack((totalpreds, treepred))

    ## Get each element's majority vote
    retpreds = []
    for i in range(nelements):
        votesEl = totalpreds[:, i]
        cnt = Counter(votesEl)
        retpreds.append(cnt.most_common(1)[0][0])

    return retpreds


'''
### calculateTreePerformanceStats
'''
def calculateTreePerformanceStats(x, y, tree, bagging=False):
    # Check for bagging
    if bagging:
        predicts = tree_pred_b(x, tree)
    else:
        predicts = tree_pred(x, tree)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for idx, el in enumerate(predicts):
        if el == 0:
            if y[idx] == 0:
                tp += 1
            else:
                fp += 1
        else:
            if el == y[idx]:
                tn += 1
            else:
                fn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / len(predicts)
    specificity = tn / (tn + fp)
    f1 = (2 * tp) / ((2 * tp) + fp + fn)
    confmatrix = [[tp, fn], [fp, tn]]
    results = {
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "specificity": specificity,
        "f1_score": f1,
        "confusion_matrix": confmatrix
    }
    return results

def impurity(array):
    cnt = Counter(array)
    keys = list(cnt.keys())
    ret = 0
    for k in keys:
        pk = cnt[k] / len(array)
        ret += pk * (1-pk)
    ret /= len(keys)
    return ret

def bestsplit(x, y):
    n = len(x)
    x, y = zip(*sorted(zip(x, y), key=lambda x: x[0]))
    x = np.array(x)
    y = np.array(y)

    candidates = (x[0:n - 1] + x[1:n]) / 2
    scores = []
    for cand in candidates:
        less = y[x <= cand]
        more = y[x > cand]
        if len(np.unique(less)) < 2:
            sless = 0
        else:
            sless = impurity(less) * (len(less)/n)
        if len(np.unique(more)) < 2:
            smore = 0
        else:
            smore = impurity(more) * (len(more)/n)
        score = (sless + smore) / 2
        scores.append(score)
    candscores = [[candidates[idx], scores[idx]]
                  for idx in range(len(candidates))]
    candscores = sorted(candscores, key=lambda x: x[1])
    return {"value": candscores[0][0], "combined-gini": candscores[0][1]}



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
    def __init__(self, values, labels, nmin, minleaf, nfeat, classnames=None):
        if classnames is None:
            classnames = [str(i) for i in range(len(labels))]
        self.root = self.produceNextNode(values, labels, nmin, minleaf, nfeat, classnames)

    def produceNextNode(self,
                        values,
                        labels,
                        nmin,
                        minleaf,
                        nfeat,
                        classes,
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
                                  columnname=classes[column],
                                  hasConstant=True,
                                  value=cutvalue)
                ret.setLeft(
                    self.produceNextNode(values[values[:, column] <= cutvalue],
                                         labels[values[:, column] <= cutvalue],
                                         nmin,
                                         minleaf,
                                         nfeat,
                                         classes=classes,
                                         parent=ret))
                ret.setRight(
                    self.produceNextNode(values[values[:, column] > cutvalue],
                                         labels[values[:, column] > cutvalue],
                                         nmin,
                                         minleaf,
                                         nfeat,
                                         classes=classes,
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
                 columnname,
                 hasConstant=True,
                 value=None,
                 secondIndex=None):
        self.parent = parent
        self.operator = operator
        self.compvalue = value
        self.hasConstant = hasConstant
        self.columnIndex = columnIndex
        self.secondIndex = secondIndex
        self.columnname = columnname

    def setLeft(self, node):
        self.left = node

    def setRight(self, node):
        self.right = node

    def print(self, tabs=0):
        return "".join(["\t" for i in range(tabs)]) + "Node. Column: " + str(
            self.columnname) + "\t- Value on split: " + str(
                self.compvalue) + "\n" + self.left.print(
                    tabs + 1) + "\n" + self.right.print(tabs + 1)


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
        return "".join(["\t" for i in range(tabs)]) + "Leaf. Elements: " + str(self.cnt)


'''
### MAIN EXECUTION OF THE ASSIGNMENT EXERCISES
'''

# Define the columns
columns = [
    'pre', 'ACD_avg', 'ACD_max', 'ACD_sum', 'FOUT_avg', 'FOUT_max', 'FOUT_sum',
    'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum',
    'NOF_avg', 'NOF_max', 'NOF_sum', 'NOI_avg', 'NOI_max', 'NOI_sum',
    'NOM_avg', 'NOM_max', 'NOM_sum', 'NOT_avg', 'NOT_max', 'NOT_sum',
    'NSF_avg', 'NSF_max', 'NSF_sum', 'NSM_avg', 'NSM_max', 'NSM_sum',
    'PAR_avg', 'PAR_max', 'PAR_sum', 'TLOC_avg', 'TLOC_max', 'TLOC_sum',
    'VG_avg', 'VG_max', 'VG_sum', 'NOCU']

# Import data for training
print("Importing training data...")
train_data = np.genfromtxt('Resources/eclipse-metrics-packages-2.0.csv',
                           delimiter=';',
                           dtype=float,
                           names=True)
classes = train_data["post"]
classes[classes > 1] = 1
train_data = train_data[columns]
tdata = np.asarray(list(train_data[0]))
for row in train_data[1:]:
    tdata = np.vstack((tdata, list(row)))

# Define parameters for growing the tree
nmin = 15
minleaf = 5
nfeat = 41


print("Training trees...")
trees = []
for cycle in tqdm(range(NUMEXECUTIONS), desc='Bagging Trees'):
    tree = tree_grow_b(tdata, classes, nmin, minleaf, nfeat, classnames=columns, m=100)
    trees.append(tree)


# Import data for testing
print("Importing testing data...")
test_data = np.genfromtxt('Resources/eclipse-metrics-packages-3.0.csv',
                          delimiter=';',
                          dtype=float,
                          names=True)
classes = test_data["post"]
classes[classes > 1] = 1
test_data = test_data[columns]
tdata = np.asarray(list(test_data[0]))
for row in test_data[1:]:
    tdata = np.vstack((tdata, list(row)))

print("Saving output...")
outputfile = "output_bagging.txt"
with open(outputfile, "w") as f:
    for idx, tree in tqdm(enumerate(trees), desc="Printing statistics"):
        f.write("Accuracy Run #" + str(idx+1) + ":\t" + str(calculateTreePerformanceStats(tdata, classes, tree, bagging=True)["accuracy"]) + "\n")
print("Done. Result printed to \"" + outputfile + "\"")
