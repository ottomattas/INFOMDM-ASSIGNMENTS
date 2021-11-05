import os, re, time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from collections import Counter
import numpy as np
import sklearn.utils
from multiprocessing import Pool
from tqdm import tqdm
from nltk import ngrams, FreqDist
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB as NaiveBayes
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score
from pattern3.es import parse as ESPparse, split as ESPsplit, tenses as ESPtenses
from googletrans import Translator

nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

VERBFEATURES = 4
NRUNS = 100

'''
### important_features(trainvectorizer, trainset,
###             trainclasses, clfclass, **parameters,
###             extrafeatures, n, nfeats)
###
### Obtains the top nfeats features for a classifier,
### averaged over nruns simulations.
###
### Parameters:
###     trainvectorizer:    Vectorizer responsible for producing n-grams
###     trainset:           Train dataset for the classifier
###     trainclasses:       Labels for each element in the dataset
###     clfclass:           sklearn Classifier with important features method
###     **parameters:       List of parameters to be fed to the classifier constructor
###     extrafeatures:      Number of extra features that have been inserted to the left
###                         of the n-gram
###     nfeats:             Number of top features to extract
###     nruns:              Number of simulations to runs
### Returns:
###     list:               Top 10 features
'''
def important_features(trainvectorizer, trainset, trainclasses, clfclass, **parameters,
                            extrafeatures=12, nfeats=10, nruns=30):
    vocab = [word for word, idx in sorted(trainvectorizer.vocabulary_.items(), key=lambda x:x[1], reverse=True)]

    matrix = []
    for i in tqdm(range(nruns)):
        clf = clfclass(parameters)
        clf.fit(trainset, trainclasses)
        matrix.append(clf.feature_importances_)


    matrix = np.matrix(matrix)
    matrix = np.add.reduce(matrix, 0)
    matrix /= nruns

    featssorted = sorted(list(enumerate(matrix.tolist()[0])), key=lambda x:x[1], reverse=True)
    outfeats = []
    for feats in featssorted:
        idx, score = feats
        newidx = idx - extrafeatures
        if newidx < 0:
            outfeats.append([idx, score])
        else:
            outfeats.append([vocab[newidx], score])

    outfeats = outfeats[:nfeats]
    return outfeats

'''
### toSpanish(text)
###
### Translates a sentence to Spanish through
### Google Translator. A delay is made after each
### iteration to prevent Google from banning the client's
### IP address for too many requests.
### 
### Parameters:
###     text: string to be translated
### Returns:
###     string: translated text
'''
def toSpanish(text):
    tr = Translator()
    out = tr.translate(text, dest='es', src='en')
    time.sleep(0.6)
    return out.text

'''
### getVerbsSpanish(text)
### 
### Obtains verb tense data from a text through
### translting it to Spanish to better detect verb
### tenses, as English has ambiguous conjugations
###
### This is not the ideal method and is very slow, but
### it was deemed acceptable after not finding a suitable
### way of finding a syntactic analyzer that could do that
### for English text.
###
### Parameters:
###     text: Text to analyze
### Returns:
###     dict: Dict containing data for number, time and person
###           of the verb, as well as the verb
'''
def getVerbsSpanish(text):
    text = toSpanish(text)
    parsed = ESPparse(text, lemmata=False, tokenize=True)

    verbs = []
    for sentence in ESPsplit(parsed):
        verbs += sentence.verbs

    ret = []
    for verb in verbs:
        verb = verb.string.split(" ")[-1]
        tenses = ESPtenses(verb)
        if tenses:
            time, person, number, tone, mode = ESPtenses(verb)[0]
            ret.append({"number":number if time != 'infinitive' and (time != 'past' and person is not None) else 0,
                    "person":person if time != 'infinitive' and (time == 'past' and person is not None) else 0,
                    "time":time, "verb":verb})
    return ret


'''
### fetchVerbFeatures(text)
###
### Gets the ratio of person and number usage in
### the verbs in text and returns it as a numpy array,
### which corresponds to the following:
###     [0]: Ratio of usage of infinitive or participle
###     [1]: Ratio of usage of first person
###     [2]: Ratio of usage of second person
###     [3]: Ratio of usage of third person
###     [4]: Ratio of usage of singular number
###     [5]: Ratio of usage of plural number
### Parameters:
###     text: Text to analyze
### Returns:
###     matrix: Numpy array of size (1,6)
'''
def fetchVerbFeatures(text):
    verbs = getVerbsSpanish(text)
    persons = Counter(map(lambda x:x["person"], verbs))
    times = Counter(map(lambda x:x["number"], verbs))
    times[0] = times.pop("singular") if "singular" in times.keys() else 0.0
    times[1] = times.pop("plural") if "plural" in times.keys() else 0.0
    verbinfo = [0 for _ in range(6)]
    for idx in persons.keys():
        verbinfo[idx] = persons[idx]
    for idx in times.keys():
        verbinfo[4+idx] = times[idx]
    total = len(verbs) if len(verbs) > 0 else 1
    verbinfo = list(map(lambda x:x/total, verbinfo))
    return verbinfo

'''
### fetchCapitalLetterRatio(text)
###
### Gets the ratio of correct capitalization for
### proper nouns and usage of the first person singular pronoun
###
### Parameters:
###     text: Text to analyze
### Returns:
###     ratiocapitalize:  Ratio of capitalization of beginning
###                       of sentences and FPS pronoun
###     ratiopropernouns: Ratio of capitalization of proper nouns
'''
def fetchCapitalLetterRatio(text):
    sentences = [x for x in text.replace("...", ".").split(".") if x is not None]
    ## Get first words of sentence
    ## and check the first letter for capital letters
    firstwords = []
    for sentence in sentences:
        words = sentence.split()
        personalI = [i for i in words if i.lower() == 'i']
        if re.match(r'^\s*[A-Za-z]+', sentence):
            firstwords.append(words[0])
        firstwords += personalI
        
        tagged_sent = pos_tag(sentence.split())
        propernouns = [word_type[0] for word_type in tagged_sent if word_type[1][:3] == 'NNP']
    
    if len(propernouns) < 1:
        ratiopropernouns = 1.0
    else:
        ratiopropernouns = sum([1 if re.match(r'^\s*[A-Z]', w) else 0 for w in propernouns])/len(propernouns)
    ratiocapitalize = sum([1 if re.match(r'^\s*[A-Z]', w) else 0 for w in firstwords])/len(firstwords)
    return [ratiocapitalize, ratiopropernouns]

'''
### applySigmoid(x)
###
### Returns the sigmoid computation for value x
###
### Parameters:
###     x: float
### Returns:
###     float: Sigmoid value of x
'''
def applySigmoid(x):
    return 1/(1 + np.exp(-x))

'''
### getTextMetaData(documents)
###
### Returns numpy matrix with 12 features obtained
### from the text in the documents passed by arguments
### The matrix is of size (ndocs, 12), and contains the
### following information:
###     [0-5]: Verb tense ratios, from fetchVerbFeatures()
###     [6]:   Ratio of capitalization of beginning of sentences and FPS pronoun
###     [7]:   Ratio of capitalization of proper nouns
###     [8]:   Positive sentiment index
###     [9]:   Negative sentiment index
###     [10]:  Neutral sentiment index
###     [11]:  Compound sentiment index
###
### Parameters:
###     documents: list of filenames
### Returns:
###     matrix: numpy matrix of size (ndocs, 12)
'''
def getTextMetaData(documents):
    ret = []
    for doc in tqdm(documents, desc="Extracting text metadata"):
        with open(doc, "r") as f:
            text = f.read()
        ## Extract verb data
        newfeats = fetchVerbFeatures(text)

        ## Fetch capitalize data
        newfeats.extend(fetchCapitalLetterRatio(text))

        ## Fetch sentiment data
        sia = SentimentIntensityAnalyzer()
        newfeats.extend(list(sia.polarity_scores(text).values()))

        ret.append(newfeats)

    ret = np.matrix(ret)
    ## Reduce sentiment compound value to range (0,1)
    ret[:,-1] = applySigmoid(ret[:,-1])

    return ret

'''
### fetchScores(clf, test, testlabels)
###
### Uses a classifier to predict the test data,
### and obtains basic statistical parameters regarding
### the predictions made.
###
### Parameters:
###     clf:        sklearn Classifier
###     test:       Matrix with test data of size (nsamples, nfeatures)
###     testlabels: List of classes corresponding to the test samples
### Returns:
###     dict: Containing accuracy, the f1 score, precision, recall,
###           specificity and the negative prediction values
'''
def fetchScores(clf, test, testlabels):
    preds = []
    preds = clf.predict(test)

    flipped_labels = [1 if x == 0 else 0 for x in testlabels]
    flipped_preds = [1 if x == 0 else 0 for x in preds]

    acc = accuracy_score(testlabels, preds)
    f1 = f1_score(testlabels, preds)
    prec = precision_score(testlabels, preds)
    recall = recall_score(testlabels, preds)
    specificity = recall_score(flipped_labels, flipped_preds)
    npv = precision_score(flipped_labels, flipped_preds)

    return {"accuracy":acc, "f1score":f1,
            "precision":prec, "negativepredvalue":npv,
            "recall":recall, "specificity":specificity}

'''
### saveBootstrapStats(clf, test, testlabels, indexes, filename)
###
### Performs a prediction test over multiple samples generated
### by the bootstrap method and writes the performance statistics
### into a file.
###
### Parameters:
###     clf:        sklearn Classifier
###     test:       Matrix with test data of size (nsamples, nfeatures)
###     testlabels: List of classes corresponding to the test samples
###     indexes:    Matrix of size (nbootstraps, nsamples) of indexes to bootstrap with
###     filename:   Name of the output file
'''
def saveBootstrapStats(clf, test, testlabels, indexes, filename=""):
    with open(filename, "w") as f:
        f.write("accuracy\tf1score\tprecision\tnegativepredvalue\trecall\tspecificity\n")
        for nextrow in tqdm(indexes, desc="Getting performance data"):
            labidx = [testlabels[idx] for idx in nextrow]
            testdocs = np.take(test, nextrow, 0)            
            score = fetchScores(clf, testdocs, labidx)
            f.write(str(round(score["accuracy"], 4)) + "\t" + str(round(score["f1score"], 4)) + "\t" +
                    str(round(score["precision"], 4)) + "\t" + str(round(score["negativepredvalue"], 4)) + "\t" +
                    str(round(score["recall"], 4)) + "\t" + str(round(score["specificity"], 4)) + "\n")

'''
### scoreBayes(trainset, train_classes, testset, test_classes,
###             sfactor_start, sfactor_end, sfactor_step)
###
### Trains several Naive-Bayes classifiers with different smoothing factors,
### fetches the best performing one, and returns the classifier,
### along with its score and smoothing factor.
###
### Parameters:
###     trainset:      Matrix with train data of size (nsamples, nfeatures)
###     trainclasses:  List of classes corresponding to the train samples
###     testset:       Matrix with test data of size (nsamples, nfeatures)
###     testclasses:   List of classes corresponding to the test samples
###     sfactor_start: Lower limit for smoothing factors
###     sfactor_end:   Upper limit for smoothing factors
###     sfactor_step:  Granularity of each smoothing factor increase
### Returns:
###     dict: Containing data for the best classifier:
###             "sfactor": Smoothing factor
###             "score": Accuracy
###             "classifier": Classifier
'''
def scoreBayes(trainset, trainclasses, testset, testclasses, sfactor_start=0.1,
                        sfactor_end=10, sfactor_step=0.1):

    ret = {'sfactor':0, 'score':0, 'classifier':None}
    ## Train Naive-Bayes
    for smoothingfactor in tqdm(np.arange(sfactor_start, sfactor_end, sfactor_step), desc="Testing different smoothing factors", leave=False):
        smoothingfactor = round(smoothingfactor, 1)
        nb = NaiveBayes(alpha=smoothingfactor)
        nb.fit(trainset, trainclasses)
        score = nb.score(testset, testclasses)
        if ret['score'] < score:
            ret = {'sfactor':smoothingfactor, 'score':score,
                'classifier':nb}
    return ret

'''
### scoreLogReg(trainset, train_classes, testset, test_classes,
###             sfactor_start, sfactor_end, sfactor_step)
###
### Trains several Logistic Regression classifiers with different
### lambda values and fetches the best performing one, and returns
### the classifier, along with its score and lambda value.
###
### Parameters:
###     trainset:      Matrix with train data of size (nsamples, nfeatures)
###     trainclasses:  List of classes corresponding to the train samples
###     testset:       Matrix with test data of size (nsamples, nfeatures)
###     testclasses:   List of classes corresponding to the test samples
###     lambda_start:  Lower limit for lambda values
###     lambda_end:    Upper limit for lambda values
###     lambda_step:   Granularity of each lambda value increase
### Returns:
###     dict: Containing data for the best classifier:
###             "lambda": Lambda value
###             "score": Accuracy
###             "classifier": Classifier
'''
def scoreLogReg(trainset, train_classes, testset, test_classes,
            lambda_start=0.1, lambda_end=10, lambda_step=0.1):

    ret = {'lambda':0, 'score':0, 'classifier':None}

    for lambd in tqdm(np.arange(lambda_start, lambda_end, lambda_step), desc="Testing different lambda values", leave=False):
        logreg = LogisticRegression(C=lambd, solver='liblinear')
        logreg.fit(trainset, train_classes)
        score = logreg.score(testset, test_classes)
        if ret['score'] < score:
            ret = {'lambda':lambd, 'score':score, 'classifier':logreg}
    return ret

'''
### scoreRandomForests(trainset, train_classes, testset, test_classes,
###             sfactor_start, sfactor_end, sfactor_step)
###
### Trains several Random Forest classifiers with different
### hyperparameter values and fetches the best performing one, and returns
### the classifier, along with its score and hyperparameters.
###
### Parameters:
###     trainset:      Matrix with train data of size (nsamples, nfeatures)
###     trainclasses:  List of classes corresponding to the train samples
###     testset:       Matrix with test data of size (nsamples, nfeatures)
###     testclasses:   List of classes corresponding to the test samples
###     ntrees_start:  Lower limit for the number of trees
###     ntrees_end:    Upper limit for lambda values
###     ntrees_step:   Granularity of each lambda value increase
###     nfeats_range:  (upperbound, lowerbound, increase)
### Returns:
###     dict: Containing data for the best classifier:
###             "lambda": Lambda value
###             "score": Accuracy
###             "classifier": Classifier
'''
def scoreRandomForests(trainset, train_classes, testset, test_classes,
            ntrees_start=50, ntrees_end=500, ntrees_step=40,
            nfeats_range=(0.05,1,0.1)):
    nfeats_start, nfeats_end, nfeats_step = nfeats_range

    ret = {'ntrees':0, 'nfeats':0, 'score':0, 'classifier':None}
    max_feats = trainset.shape[1]

    for ntrees in tqdm(np.arange(ntrees_start, ntrees_end, ntrees_step), desc="Testing different number of trees", leave=False):
        for featprop in tqdm(np.arange(nfeats_start, nfeats_end, nfeats_step), desc="Testing different number of features", leave=False):
            nfeats = round(featprop * max_feats)
            rf = RandomForestClassifier(n_estimators=ntrees, max_features=nfeats)
            rf.fit(trainset, train_classes)
            score = rf.score(testset, test_classes)
            if ret['score'] < score:
                ret = {'ntrees':ntrees, 'nfeats':nfeats, 'score':score, 'classifier':rf}
    return ret

'''
### scoreDecisionTree(trainset, train_classes, testset, test_classes,
###             sfactor_start, sfactor_end, sfactor_step)
###
### Trains several Decision Tree classifiers with different
### hyperparameter values and fetches the best performing one,
### and returns the classifier, along with its score and lambda value.
###
### Parameters:
###     trainset:      Matrix with train data of size (nsamples, nfeatures)
###     trainclasses:  List of classes corresponding to the train samples
###     testset:       Matrix with test data of size (nsamples, nfeatures)
###     testclasses:   List of classes corresponding to the test samples
###     alpha_range:   (upperbound, lowerbound, increase) values for the range
###                    of alpha values
###     maxdepth_range:(upperbound, lowerbound, increase) values for the range
###                    of maximum depth values
###     minleaf_range: (upperbound, lowerbound, increase) values for the range
###                    of values for minumum allowed leaf size
###     minsplit_range: (upperbound, lowerbound, increase) values for the range
###                    of values for minumum allowed size for node splitting
### Returns:
###     dict: Containing data for the best classifier:
###             "alpha":      Alpha value
###             "maxdepth":   Maximum depth value
###             "minleaf":    Minimum leaf size
###             "minsplit":   Minimum node size for a split
###             "score":      Accuracy
###             "classifier": Classifier
'''
def scoreDecisionTree(trainset, train_classes, testset, test_classes,
            alpha_range=(0.0,10,0.3), maxdepth_range=(5,50,3), minleaf_range=(2,30,3),
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
                    dt = DecisionTreeClassifier(max_depth=maxdepth, ccp_alpha=alpha,
                            min_samples_split=minsplit, min_samples_leaf=minleaf, max_features='log2')
                    dt.fit(trainset, train_classes)
                    score = dt.score(testset, test_classes)
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

    bootstrap_indexes = [sklearn.utils.resample(list(range(len(testdataset))), n_samples=len(testdataset), replace=True) for _ in range(len(testdataset))]
    
    print("Tokenizing the unigram features")
    trainvectorizer = CountVectorizer(input="filename", ngram_range=(1,1), strip_accents='unicode')
    unigrams = trainvectorizer.fit_transform(negafalse + negatruth)


    print("Tokenizing the unigram+bigram features")
    trainvectorizer_bi = CountVectorizer(input="filename", ngram_range=(1,2), strip_accents='unicode')    
    unibigrams = trainvectorizer_bi.fit_transform(negafalse + negatruth)
    trainclasses = [0 for x in negafalse] + [1 for x in negatruth]
    
    unigrams /= unigrams.sum(axis=1)
    unibigrams /= unibigrams.sum(axis=1)

    extrafeats = getTextMetaData(traindataset)    
    unigrams = np.hstack((extrafeats, unigrams))
    unibigrams = np.hstack((extrafeats, unibigrams))
    
    
    print("Opening and tokenizing the test documents...")
    testdocs = CountVectorizer(input="filename", ngram_range=(1,1), strip_accents='unicode', vocabulary=trainvectorizer.vocabulary_).fit_transform(testdataset)
    testdocs_bi = CountVectorizer(input="filename", ngram_range=(1,2), strip_accents='unicode', vocabulary=trainvectorizer_bi.vocabulary_).fit_transform(testdataset)
    testdocs /= testdocs.sum(axis=1)
    testdocs_bi /= testdocs_bi.sum(axis=1)

    extrafeats = getTextMetaData(testdataset)
    testdocs = np.hstack((extrafeats, testdocs))
    testdocs_bi = np.hstack((extrafeats, testdocs_bi))

    
    print("\nTraining Naïve-Bayes classifier with unigram features and obtaining hyperparameters...")
    bayes_unigram = scoreBayes(unigrams, trainclasses, testdocs, testclasses)
    saveBootstrapStats(bayes_unigram["classifier"], testdocs, testclasses, bootstrap_indexes, filename="unigram_bayes.txt")    
    print("Best Naïve-Bayes classifier for unigram features:\n" +
            "\tScore: " + str(bayes_unigram["score"]) +
            "\tSmoothing factor: " + str(bayes_unigram["sfactor"]))

    print("\nTraining Naïve-Bayes classifier with unigram+bigram features and obtaining hyperparameters...")
    bayes_unibigram = scoreBayes(unibigrams, trainclasses, testdocs_bi, testclasses)
    saveBootstrapStats(bayes_unibigram["classifier"], testdocs_bi, testclasses, bootstrap_indexes, filename="unibigram_bayes.txt")
    print("Best Naïve-Bayes classifier for unigram+bigram features:\n" +
            "\tScore: " + str(bayes_unibigram["score"]) +
            "\tSmoothing factor: " + str(bayes_unibigram["sfactor"]))
    
    
    print("\nTraining Logistic Regression with unigram features and obtaining hyperparameters...")
    logreg_unigram = scoreLogReg(unigrams, trainclasses, testdocs, testclasses,
                          lambda_start=0.001, lambda_end=1, lambda_step=0.02)
    saveBootstrapStats(logreg_unigram["classifier"], testdocs, testclasses, bootstrap_indexes, filename="unigram_logreg.txt")
    print("Best Logistic Regression classifier for unigram features:\n" +
            "\tScore: " + str(logreg_unigram["score"]) +
            "\tLambda: " + str(logreg_unigram["lambda"]))
    print("\nTraining Logistic Regression with unigram+bigram features and obtaining hyperparameters...")
    logreg_unibigram = scoreLogReg(unibigrams, trainclasses, testdocs_bi, testclasses,
                          lambda_start=0.001, lambda_end=1, lambda_step=0.02)
    saveBootstrapStats(logreg_unibigram["classifier"], testdocs_bi, testclasses, bootstrap_indexes, filename="unibigram_logreg.txt")
    print("Best Logistic Regression classifier for unigram and bigram features:\n" +
            "\tScore: " + str(logreg_unibigram["score"]) +
            "\tLambda: " + str(logreg_unibigram["lambda"]))
    

    print("\nTraining Random Forests with unigram features and obtaining hyperparameters...")
    rforest_unigram = scoreRandomForests(unigrams, trainclasses, testdocs, testclasses,
                        nfeats_range=(0.16,1,0.02))
    
    saveBootstrapStats(rforest_unigram["classifier"], testdocs, testclasses, bootstrap_indexes, filename="unigram_randomforest.txt")
    print("Best Random Forest classifier for unigram features:\n" +
            "\tScore: " + str(rforest_unigram["score"]) +
            "\tNumber of trees: " + str(rforest_unigram["ntrees"]) +
            "\tNumber of features: " + str(rforest_unigram["nfeats"]))
    print("\nTraining Random Forests with unigram+bigram features and obtaining hyperparameters...")
    rforest_unibigram = scoreRandomForests(unibigrams, trainclasses, testdocs_bi, testclasses,
                        nfeats_range=(0.03, 0.1, 0.02))
    saveBootstrapStats(rforest_unibigram["classifier"], testdocs_bi, testclasses, bootstrap_indexes, filename="unibigram_randomforest.txt")
    print("Best Random Forest classifier for unigram and bigram features:\n" +
            "\tScore: " + str(rforest_unigram["score"]) +
            "\tNumber of trees: " + str(rforest_unigram["ntrees"]) +
            "\tNumber of features: " + str(rforest_unigram["nfeats"]))

    print("\nTraining Cost-Complexity Pruning Decision tree with unigram features and obtaining hyperparameters...")
    dtree_unigram = scoreDecisionTree(unigrams, trainclasses, testdocs, testclasses)
    saveBootstrapStats(dtree_unigram["classifier"], testdocs, testclasses, bootstrap_indexes, filename="unigram_dtree.txt")
    print("Best Decision Tree classifier for unigram and bigram features:\n" +
            "\tScore: " + str(dtree_unigram["score"]) +
            "\tPruning alpha: " + str(dtree_unigram["alpha"]) +
            "\tMaximum depth of the tree: " + str(dtree_unigram["maxdepth"]) +
            "\tMinleaf: " + str(dtree_unigram["minleaf"]) +
            "\tMinsplit: " + str(dtree_unigram["minsplit"]))
    
    print("\nTraining Cost-Complexity Pruning Decision tree with unigram+bigram features and obtaining hyperparameters...")
    dtree_unibigram = scoreDecisionTree(unibigrams, trainclasses, testdocs_bi, testclasses)
    saveBootstrapStats(dtree_unibigram["classifier"], testdocs_bi, testclasses, bootstrap_indexes, filename="unibigram_dtree.txt")
    print("Best Decision Tree classifier for unigram and bigram features:\n" +
            "\tScore: " + str(dtree_unibigram["score"]) +
            "\tPruning alpha: " + str(dtree_unibigram["alpha"]) +
            "\tMaximum depth of the tree: " + str(dtree_unibigram["maxdepth"]) +
            "\tMinleaf: " + str(dtree_unibigram["minleaf"]) +
            "\tMinsplit: " + str(dtree_unibigram["minsplit"]))
