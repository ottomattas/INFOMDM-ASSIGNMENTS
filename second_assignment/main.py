import os, re
import nltk
import numpy as np
from tqdm import tqdm
from collections import defaultdict as DDict
from nltk import ngrams, FreqDist
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB as NaiveBayes


def scoreBayes(train_ngram, train_classes, test_ngrams, test_classes, sfactor_start=0.1,
                        sfactor_end=10, sfactor_step=0.1):

    ret = {'sfactor':0, 'score':0, 'classifier':None}
    ## Train Naive-Bayes
    for smoothingfactor in tqdm(np.arange(sfactor_start, sfactor_end, sfactor_step), desc="Testing different smoothing factors", leave=False):
        smoothingfactor = round(smoothingfactor, 1)
        nb = NaiveBayes(alpha=smoothingfactor)
        nb.fit(train_ngram, train_classes)
        score = nb.score(test_ngrams, test_classes)
        if ret['score'] < score:
            ret = {'sfactor':smoothingfactor, 'score':score, 'classifier':nb}
    return ret

def scoreLogReg(train_ngram, train_classes, test_ngrams, test_classes,
            lambda_start=0.1, lambda_end=10, lambda_step=0.1):

    ret = {'lambda':0, 'score':0, 'classifier':None}

    for lambd in tqdm(np.arange(lambda_start, lambda_end, lambda_step), desc="Testing different lambda values", leave=False):
        logreg = LogisticRegression(C=lambd, solver='liblinear')
        logreg.fit(train_ngram, train_classes)
        score = logreg.score(test_ngrams, test_classes)
        if ret['score'] < score:
            ret = {'lambda':lambd, 'score':score, 'classifier':logreg}
    return ret

def scoreRandomForests(train_ngram, train_classes, test_ngrams, test_classes,
            ntrees_start=50, ntrees_end=500, ntrees_step=40,
            nfeats_range=(0.05,1,0.05)):
    nfeats_start, nfeats_end, nfeats_step = nfeats_range

    ret = {'ntrees':0, 'nfeats':0, 'score':0, 'classifier':None}
    max_feats = train_ngram.shape[1]

    for ntrees in tqdm(np.arange(ntrees_start, ntrees_end, ntrees_step), desc="Testing different number of trees", leave=False):
        for featprop in tqdm(np.arange(nfeats_start, nfeats_end, nfeats_step), desc="Testing different number of features", leave=False):
            nfeats = round(featprop * max_feats)
            rf = RandomForestClassifier(n_estimators=ntrees)
            rf.fit(train_ngram, train_classes)
            score = rf.score(test_ngrams, test_classes)
            if ret['score'] < score:
                ret = {'ntrees':ntrees, 'nfeats':nfeats, 'score':score, 'classifier':rf}
    return ret

def scoreDecisionTree(train_ngram, train_classes, test_ngrams, test_classes,
            alpha_range=(0.0,10,0.1), maxdepth_range=(5,50,1), minleaf_range=(2,30,2),
                minsplit_range=(5,40,3)):

    alpha_start, alpha_end, alpha_step = alpha_range
    maxdepth_start, maxdepth_end, maxdepth_step = maxdepth_range
    minleaf_start, minleaf_end, minleaf_step = minleaf_range
    minsplit_start, minsplit_end, minsplit_step = minsplit_range

    ret = {'alpha':0, 'maxdepth':0, 'score':0, 'classifier':None, 'minleaf':0, 'minsplit':0}
    for maxdepth in tqdm(range(maxdepth_start, maxdepth_end, maxdepth_step), desc="Testing different tree depths", leave=False):
        for alpha in tqdm(np.arange(alpha_start, alpha_end, alpha_step), desc="Testing different alpha values", leave=False):
            for minleaf in range(minleaf_start, minleaf_end, minleaf_step):
                for minsplit in range(minsplit_start, minsplit_end, minsplit_step):
                    dt = DecisionTreeClassifier(max_depth=maxdepth, ccp_alpha=alpha, max_features='auto',
                            min_samples_split=minsplit, min_samples_leaf=minleaf)
                    dt.fit(train_ngram, train_classes)
                    score = dt.score(test_ngrams, test_classes)
                    if ret['score'] < score:
                        ret = {'alpha':alpha, 'maxdepth':maxdepth, 'score':score,
                                    'classifier':dt, 'minleaf':minleaf, 'minsplit':minsplit}
    return ret

if __name__ == '__main__':
    files = [str(x) for x in Path('Resources' + os.path.sep + 'op_spam_v1.4').rglob('*.txt')
                        if 'negative' in str(x)]
    traindataset = [x for x in files if 'fold5' not in x]
    testdataset = [x for x in files if 'fold5' in x]
    testclasses = [0 if 'deceptive' in x else 1 for x in testdataset]
    negatruth = [x for x in traindataset if 'truthful' in x]
    negafalse = [x for x in traindataset if 'deceptive' in x]

    #positruth = [x for x in traindataset if 'positive' in x and 'truthful' in x]
    #posifalse = [x for x in traindataset if 'positive' in x and 'deceptive' in x]
    print("Tokenizing the unigram features")
    trainvectorizer = CountVectorizer(input="filename", ngram_range=(1,1))
    unigrams = trainvectorizer.fit_transform(negafalse + negatruth)
    print("Tokenizing the unigram+bigram features")
    trainvectorizer_bi = CountVectorizer(input="filename", ngram_range=(1,2))
    unibigrams = trainvectorizer_bi.fit_transform(negafalse + negatruth)
    trainclasses = [0 for x in negafalse] + [1 for x in negatruth]
    
    print("Opening and tokenizing the test documents...")
    testdocs = CountVectorizer(input="filename", ngram_range=(1,1), vocabulary=trainvectorizer.vocabulary_).fit_transform(testdataset)
    testdocs_bi = CountVectorizer(input="filename", ngram_range=(1,2), vocabulary=trainvectorizer_bi.vocabulary_).fit_transform(testdataset)

    
    print("\nTraining Naïve-Bayes classifier with unigram features and obtaining hyperparameters...")
    bayes_unigram = scoreBayes(unigrams, trainclasses, testdocs, testclasses)
    print("Best Naïve-Bayes classifier for unigram features:\n" +
            "\tScore: " + str(bayes_unigram["score"]) +
            "\tSmoothing factor: " + str(bayes_unigram["sfactor"]))
    print("\nTraining Naïve-Bayes classifier with unigram+bigram features and obtaining hyperparameters...")
    bayes_unibigram = scoreBayes(unibigrams, trainclasses, testdocs_bi, testclasses)
    print("Best Naïve-Bayes classifier for unigram+bigram features:\n" +
            "\tScore: " + str(bayes_unibigram["score"]) +
            "\tSmoothing factor: " + str(bayes_unibigram["sfactor"]))
    
    
    print("\nTraining Logistic Regression with unigram features and obtaining hyperparameters...")
    logreg_unigram = scoreLogReg(unigrams, trainclasses, testdocs, testclasses,
                          lambda_start=0.001, lambda_end=1, lambda_step=0.02)
    print("Best Logistic Regression classifier for unigram features:\n" +
            "\tScore: " + str(logreg_unigram["score"]) +
            "\tLambda: " + str(logreg_unigram["lambda"]))
    print("\nTraining Logistic Regression with unigram+bigram features and obtaining hyperparameters...")
    logreg_unibigram = scoreLogReg(unibigrams, trainclasses, testdocs_bi, testclasses,
                          lambda_start=0.001, lambda_end=1, lambda_step=0.02)
    print("Best Logistic Regression classifier for unigram and bigram features:\n" +
            "\tScore: " + str(logreg_unibigram["score"]) +
            "\tLambda: " + str(logreg_unibigram["lambda"]))

    print("\nTraining Random Forests with unigram features and obtaining hyperparameters...")
    rforest_unigram = scoreRandomForests(unigrams, trainclasses, testdocs, testclasses,
                        nfeats_range=(0.16,1,0.02))
    print("Best Random Forest classifier for unigram features:\n" +
            "\tScore: " + str(rforest_unigram["score"]) +
            "\tNumber of trees: " + str(rforest_unigram["ntrees"]) +
            "\tNumber of features: " + str(rforest_unigram["nfeats"]))
    print("\nTraining Random Forests with unigram+bigram features and obtaining hyperparameters...")
    rforest_unibigram = scoreRandomForests(unibigrams, trainclasses, testdocs_bi, testclasses,
                        nfeats_range=(0.03, 0.1, 0.02))
    print("Best Random Forest classifier for unigram and bigram features:\n" +
            "\tScore: " + str(rforest_unigram["score"]) +
            "\tNumber of trees: " + str(rforest_unigram["ntrees"]) +
            "\tNumber of features: " + str(rforest_unigram["nfeats"]))

    print("\nTraining Cost-Complexity Pruning Decision tree with unigram features and obtaining hyperparameters...")
    dtree_unigram = scoreDecisionTree(unigrams, trainclasses, testdocs, testclasses)
    print("Best Decision Tree classifier for unigram and bigram features:\n" +
            "\tScore: " + str(dtree_unigram["score"]) +
            "\tPruning alpha: " + str(dtree_unigram["alpha"]) +
            "\tMaximum depth of the tree: " + str(dtree_unigram["maxdepth"]) +
            "\tMinleaf: " + str(dtree_unigram["minleaf"]) +
            "\tMinsplit: " + str(dtree_unigram["minsplit"]))
    
    print("\nTraining Cost-Complexity Pruning Decision tree with unigram+bigram features and obtaining hyperparameters...")
    dtree_unibigram = scoreDecisionTree(unigrams, trainclasses, testdocs, testclasses)
    print("Best Decision Tree classifier for unigram and bigram features:\n" +
            "\tScore: " + str(dtree_unibigram["score"]) +
            "\tPruning alpha: " + str(dtree_unibigram["alpha"]) +
            "\tMaximum depth of the tree: " + str(dtree_unibigram["maxdepth"]) +
            "\tMinleaf: " + str(dtree_unibigram["minleaf"]) +
            "\tMinsplit: " + str(dtree_unibigram["minsplit"]))


















