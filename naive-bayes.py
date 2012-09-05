#! /usr/bin/env python
import sys
import math

"""
This programme implements a Naive Bayes classifier(multinominal) applied to text categorizaton.
It is used as a comparasion to Naive Bayes with Gibbs sampling.

Reference:
@book{manning2008introduction,
  title={Introduction to information retrieval},
  author={Manning, C.D. and Raghavan, P. and Schutze, H.},
  volume={1},
  year={2008},
  publisher={Cambridge University Press Cambridge}
}
"""
NUM_CATEGORY = 2 # number of categories
category_map = dict({"course":0, "faculty":1})

magic_key_set = set(["_CATEGORY", "_NUM_WORDS"])

# print out docditct
def print_dict(d):
    it = d.iteritems()
    try:
        while 1:
            (key, value) = it.next()
            if key in magic_key_set:
                print key + " --> " + str(value)
            else:
                print key + " --> [ " + str(value[0]) + " , " + str(value[1]) + " ]"
    except StopIteration:
        pass

# extracting tokens from doc and build a vocabulary
def extract_token(line):
    docdict = dict()
    line = line.replace("\t", " ") # raplace \t
    line = line.replace("\n", "") # remove \n
    line = line.split(" "); # tokenize the document
    
    word_count = 0 # count words in doc
    if line[0] == "course" or line[0] == "faculty":
        category = category_map[line[0]]
        # add words(terms) to doc
        del line[0]
        for word in line:
            if word not in docdict:
                # dict: strword --> [index, count], index is unknown here for we don't have a vocabulary yet
                docdict[word] = [-1, 0] 
            value = docdict[word]
            value[1] = value[1] + 1 # increase term count
            word_count = word_count + 1
            docdict[word] = value
    else:
        category = -1 # unknown category
    
    # assign category info to every doc, a bit hacking here
    docdict["_CATEGORY"] = category
    docdict["_NUM_WORDS"] = word_count
    return (docdict, category)
    
# build vocabulary
def build_vocabulary(docdict, vocabulary):
    it = docdict.iteritems()
    try:
        while 1:
            (key, value) = it.next()
            # omit magic keys
            if key in magic_key_set:
                continue
            if key not in vocabulary:
                index = len(vocabulary)
                vocabulary[key] = [index, 0]

            v = vocabulary[key]; v[1] = v[1] + value[1]
            index = v[0]
            vocabulary[key] = v
            docdict[key] = [index, value[1]]
    except StopIteration:
        pass

# build terms for each category using every doc    
def build_cterm(docdict, vocabulary, cterm):
    it = docdict.iteritems()
    try:
        while 1:
            (key, value) = it.next()
            # omit magic keys
            if key in magic_key_set:
                continue
            if key not in cterm:
                index = vocabulary[key][0]
                cterm[key] = [index, 0]
            cv = cterm[key]; cv[1] = cv[1] + value[1];
            cterm[key] = cv
    except StopIteration:
        pass
    except KeyError:
        print "Word '" + key + "' not found in vocabulary!"
        exit()
    # add total words for each category
    cterm["_NUM_WORDS"] = cterm["_NUM_WORDS"] + docdict["_NUM_WORDS"]

# train a multinomial Naive Bayes classifier
def train_multinomial_NB(train_file_path):
    trainfile = open(train_file_path, "r")
    vocabulary = dict({"a":[0,0]}) # vocabulary for the whole dataset
    doclist = []
    Nc = [0] * NUM_CATEGORY # number of docs in every class
    cterm_list = [] # counts terms in each category
    for i in range(NUM_CATEGORY):
        cterm_list.append(dict({"_NUM_WORDS":0}))
        
    for line in trainfile:
        (docdict, category) = extract_token(line)
        if category == -1: # omit other categories
            continue
        Nc[category] = Nc[category] + 1
        build_vocabulary(docdict, vocabulary)
        build_cterm(docdict, vocabulary, cterm_list[category])
        doclist.append(docdict)
        
    print "Vocabulary size: " + str(len(vocabulary))
    print "Number of docs:  " + str(len(doclist))
    #print_dict(vocabulary)
    #print_dict(doclist[0])
    prior = [0] * NUM_CATEGORY # priors for each class
    condprob = [] # conditional prob condprob[c][t]
    assert(len(doclist) == sum(Nc))
    N = sum(Nc) # total number of docs
    V = len(vocabulary) # size of vocabulary
    for c in range(NUM_CATEGORY):
        prior[c] = Nc[c] * 1.0 / N
        p = [-1] * V
        cterm = cterm_list[c]
        it = cterm.iteritems()
        total_words = cterm["_NUM_WORDS"] # total words in each category
        try:
            while 1:
                (key, value) = it.next()
                if key in magic_key_set:
                    continue
                index = value[0]
                count = value[1]
                p[index] = (count + 1) * 1.0 / (total_words + V)
        except StopIteration:
            pass
        # assign conditional prob that dose not exist in cterm
        for i in range(V):
            if p[i] == -1:
                p[i] = 1.0 / (total_words + V)
        assert(abs(sum(p) - 1) < 0.000001)
        condprob.append(p)
        
    return (doclist, vocabulary, prior, condprob)
    
# apply multinomial Naive Bayes classifier to docs
def apply_multinomial_NB(prior, condprob, docdict):
    it = docdict.iteritems()
    score = [0] * NUM_CATEGORY
    for c in range(NUM_CATEGORY):
        score[c] = prior[c]
    try:
        while 1:
            (key, value) = it.next()
            if key in magic_key_set:
                continue
            index = value[0]
            count = value[1]
            for c in range(NUM_CATEGORY):
                score[c] = score[c] + count * math.log(condprob[c][index])
    except StopIteration:
        pass
        
    max_score = -float("inf")
    cat = -1
    for c in range(NUM_CATEGORY):
        if score[c] > max_score:
            cat = c
            max_score = score[c]
            
    return cat

# assgin word index for test documents
def assign_word_index(docdict, vocabulary):
    it = docdict.iteritems()
    unknown_words = []
    try:
        while 1:
            (key, value) = it.next()
            if key in magic_key_set:
                continue
            if key not in vocabulary:
                unknown_words.append(key)
                continue
            index = vocabulary[key][0]
            value[0] = index
            docdict[key] = value
    except StopIteration:
        pass
        
    # remove unknown words, i.e. keys that don't exist in vocabulary
    for word in unknown_words:
        del docdict[word]
            
# Note: test_file_path contains multiple test documents
def classify_test_documents(prior, condprob, vocabulary, test_file_path):
    testfile = open(test_file_path, "r")
    testdoc_list = []
    predictions = []
    for line in testfile:
        (docdict, category) = extract_token(line)
        if category == -1:
            continue
        # assgin word index for test documents
        assign_word_index(docdict, vocabulary)
        testdoc_list.append(docdict)
        cat = apply_multinomial_NB(prior, condprob, docdict)
        predictions.append(cat)
        
    return (predictions, testdoc_list)

# evaluate predictions    
def evaluate_classification(predictions, labels):
    assert(len(predictions) == len(labels))
    n = len(labels)
    correct = 0
    for i in range(n):
        if predictions[i] == labels[i]:
            correct = correct + 1
    
    accuracy = correct * 1.0 / n
    print "Number of test documents: " + str(n)
    print str(correct) + " of " + str(n) + " documents are correctly classified."
    print "Accuracy: " + str(accuracy)
    
def main(argv):
    print "Usage: python ./naive-bayes.py training_docs testing_docs"
    # training
    (doclist, vocabulary, prior, condprob) = train_multinomial_NB(argv[1])
    # apply Naive bayes classifier
    (predictions, testdoc_list) = classify_test_documents(prior, condprob, vocabulary, argv[2])
    
    labels = []
    for doc in testdoc_list:
        cat = doc["_CATEGORY"]
        labels.append(cat)
    evaluate_classification(predictions, labels)
    
    
if __name__ == "__main__":
    main(sys.argv)
