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


def tree_grow(x, y, nmin, minleaf, nfeat=None):
    if nfeat is None:
        nfeat = len(x)
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


def tree_grow_b(x, y, nmin, minleaf, nfeat=None, m=1):
    if nfeat is None:
        nfeat = x.shape[0]
    ret = []
    for i in range(m):
        indexes = np.random.choice(range(x.shape[0]), size=x.shape[0])
        ret.append(
            tree_grow(x[indexes, :], y[indexes, :], nmin, minleaf, nfeat))
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
        predictions.append(pred)

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


def calculateTreePerformanceStats(x, y, tree):
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


'''
nmin = 20
minleaf = 5
nfeat = 0
pima_data = np.genfromtxt('pima.txt', delimiter=',')
tree = tree_grow(pima_data[:, list(range(0,8))], pima_data[:, -1], nmin, minleaf,
                 nfeat)
print(calculateTreePerformanceStats(pima_data[:, list(range(0,8))], pima_data[:, -1], tree))


nmin = 2
minleaf = 1
nfeat = 0
tree = tree_grow(credit_data[:, range(0, 5)], credit_data[:, 5], nmin, minleaf,
                 nfeat)
print(tree)

'''
columns = [
    'pre', 'ACD_avg', 'ACD_max', 'ACD_sum', 'FOUT_avg', 'FOUT_max', 'FOUT_sum',
    'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum',
    'NOF_avg', 'NOF_max', 'NOF_sum', 'NOI_avg', 'NOI_max', 'NOI_sum',
    'NOM_avg', 'NOM_max', 'NOM_sum', 'NOT_avg', 'NOT_max', 'NOT_sum',
    'NSF_avg', 'NSF_max', 'NSF_sum', 'NSM_avg', 'NSM_max', 'NSM_sum',
    'PAR_avg', 'PAR_max', 'PAR_sum', 'TLOC_avg', 'TLOC_max', 'TLOC_sum',
    'VG_avg', 'VG_max', 'VG_sum', 'NOCU'
]

print("Importing data...")
train_data = np.genfromtxt('eclipse-metrics-packages-2.0.csv',
                           delimiter=';',
                           dtype=float,
                           names=True)
classes = train_data["post"]
classes[classes > 1] = 1
train_data = train_data[columns]
tdata = np.asarray(list(train_data[0]))
for row in train_data[1:]:
    tdata = np.vstack((tdata, list(row)))

nmin = 15
minleaf = 5
nfeat = 41
m = 100
print("Growing tree...")
#print(train_data.dtype.names[32])
tree = tree_grow(tdata, classes, nmin, minleaf, nfeat)
print(tree)

test_data = np.genfromtxt('eclipse-metrics-packages-3.0.csv',
                          delimiter=';',
                          dtype=float,
                          names=True)
classes = test_data["post"]
classes[classes > 1] = 1
test_data = test_data[columns]
tdata = np.asarray(list(train_data[0]))
for row in train_data[1:]:
    tdata = np.vstack((tdata, list(row)))
