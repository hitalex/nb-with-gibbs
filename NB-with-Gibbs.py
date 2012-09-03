# -*- coding: utf-8 -*-

#! /usr/bin/env python
import sys
from math import log
from math import exp
import random

from Dirichlet import Dirichlet

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
vocabulary = dict({"a":(0,0)}) # vocabulary for the whole dataset

# Add words in vocabulary
def add_in_dict(line):
    doc = line[1:-1]
    for word in doc:
        try:
            value = vocabulary[word]
        except KeyError:
            vocabulary[word] = (len(vocabulary), 0)
        finally:
            value = vocabulary[word]
            value[1] = value[1] + 1
            vocabulary[word] = value

"""
Create index for every document
Every docuement is associated with a dict, which is defined as str --> (index, count), 
where str refers to the word, index refers to index in vocabulary, count refers to 
total count in this document

@return the map for this document
"""
def create_index(line):
    docdict = dict()
    for word in line:
        try:
            value = vocabulary[word]
        except KeyError:
            print "Error: word not found in vocabulary!"
            exit()
        
        index = value[0]
        try:
            value = docdict[word]
        except KeyError:
            docdict[word] = (index, 0)
        finally:
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
Caculate Pr for each class
@count documents for a class
@N number of all the documents
@index index of the document to be processed
@gamma_para Gamma parameters
@theta theta parameters for each class
"""    
def caculate_Pr(c, class_count, N, index, docindexset, gamma_para, theta):
    ln_Pr = log(class_count[c] + gamma_para[c] - 1) - log(N + gamma_para[0] + gamma_para[1] - 1)
    items = docindex[index]
    for (word, t) in items:
        ln_Pr = ln_Pr + t[1] * log(theta[t[0]])
        
    return ln_Pr
    
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
    class_count = (0, 0) # number of documents in each class
    for i in range(N):
        labels[i] = random.randint(0, 1)
        class_count[labels[i]] = class_count[labels[i]] + 1
        
    gamma_para = (1, 1) # initial values for Beta distribution
    alpha = [1] * V # hyper parameter for the parameter \theta 
    # theta0 and theta1 are the multinominal prior for words
    theta0 = Dirichlet(alpha)
    theta1 = Dirichlet(alpha)
    
    for i in range(T):
        for j in range(N):
            label = labels[j]
            assert(label == 0 or label == 1)
            class_count[label] = class_count[label] - 1
            ln_Pr0 = caculate_Pr(0, class_count, N, j, docindexset, gamma_para, theta0)
            ln_Pr1 = caculate_Pr(1, class_count, N, j, docindexset, gamma_para, theta1)
            ratio = exp(ln_Pr0 - ln_Pr1)
            
    
    
if __name__ == "__main__":
    main(sys.argv)
