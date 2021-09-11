import numpy as np
from collections import Counter
from ClassificationTreeDM import OpType as op, ClassificationTreeDM as tdm

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
## TODO
def tree_grow(x, y, nmin, minleaf, nfeat):
    pass


## TODO
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    pass

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
        if totalpreds = None:
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
