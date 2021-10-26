# 1. Overview of stuff to do
#### Performance
Currently we have pretty decent performance on most classifiers, but we could definitely improve it, specially for the decision tree one, which currently has a pretty disappointing performance (73%)

#### Extra functions needed for analysis
Right now the code extracts accuracy of the classifiers over the test data, but not recall, f1-scores, and such. Extracting them should be done to present the analysis data.

#### Further testing
As per the assignment, we are operating on the 4 first folds for training, and on the last for testing. We could for example try several different fold combinations, to try to find the overall best performing hyperparameters.

# 2. What we have now:
We are currently using n-grams for the algorithms, so mostly basing them on word frequency across all samples. Furthermore, we use ALL words found in the training data, instead of a portion of them (like the X most frequent words, for example). This might prove ideal for some algorithms like Naïve-Bayes classifiers, but it may not be the ideal form of data to be fed for some other classifiers.

# 3. Some alternatives
#### Smaller vocabulary
As previously mentioned, we can diminish the vocabulary found in training to reduce the amount of attributes. I don't know if this will increase performance in any way, but it will definitely speed up the training process by some amount.

#### Normalizing word count
Currently the program is dealing with raw word counts. It is a common practice to normalize the data to reduce the mean and standard deviation, as well as eliminate some types of outliers. I do not know if it will be of help, but definitely worth a try.

### Other types of attributes
#### Word functions
NLTK is a library to deal with text processing, and has an analyzer to detect sentence functions within words of plain text. This type of count might be a source of useful information to analyze the text, and maybe provide a different type of insight.

#### Metadata for the text
Review character length, number of sentences, average sentence length..., etc. These can be several added parameters that may or may not help some classifiers perform better. We can try these to see if they shake up the performance, for example.

#### Idk, anything else you guys can come up with? :P

difficulties faced by liars in encoding spatial information
a plausible relationship between deceptive opinion spam and imaginative writing, based on POS distributional similarities
the top 5 highest weighted features for each class
The SVM cost parameter, C, is tuned by nested cross-validation on the training data


three types of language features: 
(1) changes in first-person singular use, often attributed to psychological distancing (Newman et al., 2003), 
(2) decreased spatial awareness and more narrative form, consistent with theories of reality monitoring (Johnson and Raye, 1981) and imaginative writing (Biber et al., 1999; Rayson et al., 2001), more verbs relative to nouns than truthful
(3) increased negative emotion terms, often attributed to leakage cues (Ekman and Friesen, 1969), but perhaps better explained in our case as an exaggeration of the underlying review sentiment.
