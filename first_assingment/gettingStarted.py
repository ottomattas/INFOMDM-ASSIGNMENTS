import numpy as np
import math
from collections import Counter

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)


def impurity(array):
    cnt = Counter(array)
    keys = list(cnt.keys())
    ret = 0
    for k in keys:
        pk = cnt[k] / len(array)
        ret += pk * (1-pk)
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


'''print("Impurity for practice excercise:")
print(impurity([1,0,1,1,1,0,0,1,1,0,1]))
print("")

print("Best-Split for practice excercise:")
print(bestsplit(credit_data[:,3], credit_data[:,5]))
'''
