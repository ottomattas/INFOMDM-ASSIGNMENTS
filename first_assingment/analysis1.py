from main import *

# Define the columns
columns = [
    'pre', 'ACD_avg', 'ACD_max', 'ACD_sum', 'FOUT_avg', 'FOUT_max', 'FOUT_sum',
    'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum',
    'NOF_avg', 'NOF_max', 'NOF_sum', 'NOI_avg', 'NOI_max', 'NOI_sum',
    'NOM_avg', 'NOM_max', 'NOM_sum', 'NOT_avg', 'NOT_max', 'NOT_sum',
    'NSF_avg', 'NSF_max', 'NSF_sum', 'NSM_avg', 'NSM_max', 'NSM_sum',
    'PAR_avg', 'PAR_max', 'PAR_sum', 'TLOC_avg', 'TLOC_max', 'TLOC_sum',
    'VG_avg', 'VG_max', 'VG_sum', 'NOCU'
]

# Import data for training
print("Importing training data...")
train_data = np.genfromtxt('Rescources/eclipse-metrics-packages-2.0.csv',
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

# Grow a tree
print("Growing tree...")
tree = tree_grow(tdata, classes, nmin, minleaf, nfeat, classnames=columns)

# Print the tree for debugging
print(tree)

# Import data for training
print("Importing testing data...")
test_data = np.genfromtxt('eclipse-metrics-packages-3.0.csv',
                          delimiter=';',
                          dtype=float,
                          names=True)
classes = test_data["post"]
classes[classes > 1] = 1
test_data = test_data[columns]
tdata = np.asarray(list(test_data[0]))
for row in test_data[1:]:
    tdata = np.vstack((tdata, list(row)))
print(calculateTreePerformanceStats(tdata, classes, tree, bagging=False))
