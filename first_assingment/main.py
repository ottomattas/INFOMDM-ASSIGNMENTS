import numpy as np
from tqdm import tqdm
from collections import Counter
from ClassificationTreeDM import OpType as op, ClassificationTreeDM

# Define input data
credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

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

# Define a function for growing a tree for bagging
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
    nelements = len(x[0])
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
