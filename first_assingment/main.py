import numpy as np
from collections import Counter
from ClassificationTreeDM import OpType as op, ClassificationTreeDM
credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

'''
### tree_grow()
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
def tree_grow(x, y, nmin, minleaf, nfeat=len(x)):
    return ClassificationTreeDM(x, y, nmin, minleaf, nfeat)


'''
### tree_grow_b()
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
def tree_grow_b(x, y, nmin, minleaf, nfeat=x.shape[0], m):
    ret = []
    for i in range(m):
        indexes = np.random.choice(range(x.shape[0]), size=x.shape[0])
        ret.append(tree_grow(x[indexes,:], y[indexes,:], nmin, minleaf, nfeat))
    return ret

'''
### tree_pred()
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
        predictions.append(row)

    return predictions

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
    nelements = len(x[0])
    totalpreds = None

    ## Obtain each prediction and put them into a
    ## numpy matrix, each row is a tree's prediction list
    for tree in trees:
        treepred = numpy.array(tree_pred(x, tree))
        if totalpreds is None:
            totalpreds = treepred
        else:
            totalpreds = np.vstack((totalpreds, treepred))

    ## Get each element's majority vote
    retpreds = []
    for i in range(nelements):
        votesEl = totalpreds[:,i]
        cnt = Counter(votesEl)
        retpreds.append(cnt.most_common(1)[0][0])

    return retpreds

nmin = 2
minleaf = 1
nfeat = 0
tree = tree_grow(credit_data[:,range(0, 5)], credit_data[:,5], nmin, minleaf, nfeat)
print(tree.print())
