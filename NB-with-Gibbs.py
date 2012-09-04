# -*- coding: utf-8 -*-

#! /usr/bin/env python
import sys
from math import log
from math import exp
import random

from utils import Dirichlet, choose

"""
This programme deals with Gibbs sampling applied to a Naive Bayes classifier.
Note: Only binary classification is supported.

Reference: 
Bibtex: @techreport{resnik2010gibbs,
  title={Gibbs sampling for the uninitiated},
  author={Resnik, P. and Hardisty, E.},
  year={2010},
  institution={DTIC Document}
}

"""

"""
Note: vocabulary is a mapping from str to a tuple, in which, the first elements is the word index, 
and the seconde is the word count
This means if you want find a word with a specified index , you have to iterate the whole dict, 
and it becomes much simpler when you have word in string form, like "mini"
"""
vocabulary = dict({"a":[0,0]}) # vocabulary for the whole dataset

# Add words in vocabulary
def add_in_dict(line):
    doc = line[1:-1]
    for word in doc:
        if word not in vocabulary:
            vocabulary[word] = [len(vocabulary), 0]
        value = vocabulary[word]
        value[1] = value[1] + 1
        vocabulary[word] = value

"""
Create index for every document
Every docuement is associated with a dict, which is defined as str --> [index, count], 
where str refers to the word, index refers to index in vocabulary, count refers to 
total count in this document

@return the map for this document
"""
def create_index(line):
    doc = line[1:-1]
    docdict = dict()
    for word in doc:
        try:
            value = vocabulary[word]
        except KeyError:
            print "Error: word \"" + word + "\" not found in vocabulary!"
            exit()
        
        index = value[0]
        if word not in docdict:
            docdict[word] = [index, 0]
        value = docdict[word]
        value[1] = value[1] + 1
        docdict[word] = value
    
    return docdict

"""
Load data from a file given the full file path @datafile
Note: Only load two categories: course and faculty.    
"""
def load_data(datafile):
    docset = []
    docindexset = []
    f = open(datafile, "r")
    for line in f:  # one document in each line
        line = line.replace("\t", " ")
        line = line.split(" "); # tokenize the document
        del line[-1] # delete the \n
        print "Line[0]:" + line[0]
        if line[0] == "course" or line[0] == "faculty":
            add_in_dict(line) # add words in vocabulary
            docindex = create_index(line) # create index for every word in every document
            docset.append(line)
            docindexset.append(docindex)
    #print docset
    return (docset, docindexset)

"""
Count word for each class
"""
def count_word(docindexset, labels, word_count):
    N = len(labels)
    for i in range(N): # for every document
        label = labels[i]
        doc = docindexset[i]
        it = doc.iteritems()
        try:
            while 1:
                (key, value) = it.next()
                index = value[0]
                word_count[label][index] = word_count[label][index] + value[1]
        except StopIteration:
            pass
        
"""
Caculate Pr for each class
@count documents for a class
@N number of all the documents
@index index of the document to be processed
@hyper_gamma hyperparameter for Gamma distribution
@theta theta parameters for each class
"""    
def caculate_Pr(c, class_count, N, index, docindexset, hyper_gamma, theta):
    ln_Pr = log(class_count[c] + hyper_gamma[c] - 1) - log(N + hyper_gamma[0] + hyper_gamma[1] - 1)
    docindex = docindexset[index]
    it = docindex.iteritems()
    try:
        while 1:
            (word, t) = it.next()
            ln_Pr = ln_Pr + t[1] * log(theta[t[0]])
    except StopIteration:
        pass

    return ln_Pr
    
"""
Update word count for each class by removing or addint document j
@docdict the dict for the document j
@c which class
@sign indicate whether remove(-1) or add(+1)
@word_count the word count for each class
"""
def update_word_count(docdict, c, sign, word_count):
    it = docdict.iteritems()
    try:
        while 1:
            (key, value) = it.next()
            index = value[0]
            word_count[c][index] = word_count[c][index] + sign * value[1]
    except StopIteration:
        pass
"""
Update theta, the word distribution for two classes
theta_x ~ Dirichlet(t_x)
"""    
def update_theta(docindexset, labels, theta, hyper_multi, word_count):
    N = len(labels) # number of documents
    V = len(vocabulary) # size of vocabulary
    for c in (0, 1): # for class 0 and class 1
        t = [0] * V
        for i in range(V):
            t[i] = word_count[c][i] + hyper_multi[i]
        theta[c] = Dirichlet(t)
    
# main loop    
def main(argv):
#    print "File to open: " + argv[1]
    print "Usage: python ./NB-with-Gibbs.py data_file max_iters"
    (docset, docindexset) = load_data(argv[1])
    
    # initializations for parameters
    T = int(argv[2]) # max iterations
    N = len(docset) # number of documents
    V = len(vocabulary) # size of the vocabulary
    
    #randomly set labels for documents
    labels = [0] * N # labes for all documents, 0 or 1
    class_count = [0, 0] # number of documents in each class
    for i in range(N):
        labels[i] = random.randint(0, 1)
        class_count[labels[i]] = class_count[labels[i]] + 1
        
    # count word in each class
    word_count = [[0]*V, [0]*V]
    count_word(docindexset, labels, word_count)
    
    hyper_gamma = (2, 2) # parameter for Beta distribution
    hyper_multi = [1] * V # hyperparameter vector for the multinominal prior
    # theta0 and theta1 are the multinominal prior for words
    theta0 = Dirichlet(hyper_multi)
    theta1 = Dirichlet(hyper_multi)
    theta = [theta0, theta1]
    
    print "Start to iterate..."
    for i in range(T):
        for j in range(N):
            label = labels[j]
            assert(label == 0 or label == 1)
            class_count[label] = class_count[label] - 1
            update_word_count(docindexset[j], label, -1, word_count) # update word count by removing the docuemnt j
            
            ln_Pr0 = caculate_Pr(0, class_count, N, j, docindexset, hyper_gamma, theta[0])
            ln_Pr1 = caculate_Pr(1, class_count, N, j, docindexset, hyper_gamma, theta[1])
            ratio = exp(ln_Pr0 - ln_Pr1)
            index = choose((0, 1), [ratio, 1]) # choose according a distribution
            new_label = (0, 1)[index]
            
            labels[j] = new_label
            class_count[new_label] = class_count[new_label] + 1
            update_word_count(docindexset[j], new_label, +1, word_count) # update word count by adding the docuemnt j
        
        update_theta(docindexset, labels, theta, hyper_multi, word_count)
        
    print "Iteration done."
    print "Results - Labels of each document:"
    print labels
    
if __name__ == "__main__":
    main(sys.argv)
