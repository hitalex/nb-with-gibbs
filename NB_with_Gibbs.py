
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
used_category = ["acq", "alum"] # category used for classification

vocabulary = dict({"a":[0,0]}) # vocabulary for the whole dataset
class_map = dict({used_category[0]:0, used_category[1]:1})
    
logger = open("logger", "w")
output_file = logger # or output_file = sys.stdout

# Add words in vocabulary
def add_in_dict(line):
    doc = line[1: len(line)]
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
    doc = line[1 : len(line)]
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
Note: Only load two categories: used_category[0] and used_category[1]
"""
def load_data(datafile):
    docset = []
    docindexset = []
    docnum = [0, 0]
    f = open(datafile, "r")
    for line in f:  # one document in each line
        line = line.replace("\t", " ") # raplace \t
        line = line.replace("\n", "") # remove \n
        line = line.split(" "); # tokenize the document
        #print "Line[0]:" + line[0]
        if line[0] == used_category[0] or line[0] == used_category[1]:
            cindex = class_map[line[0]]
            docnum[cindex] = docnum[cindex] + 1
            add_in_dict(line) # add words in vocabulary
            docindex = create_index(line) # create index for every word in every document
            docset.append(line)
            docindexset.append(docindex)
    #print docset
    return (docset, docindexset, tuple(docnum))

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
        
"""
Evaluate the clustering result
@docnum number of docs in every class
methods to be used: purity, NML, RI, F-value
"""
def evaluate_cluster_result(labels, docset, N, docnum):
    print "Clustering evaluation result:"
    for i in range(N):
        doc = docset[i]
        count0 = 0 # count number of used_category[0] docs in cluster 0
        count1 = 0 # count number of used_category[1] docs in cluster 0
        if labels[i] == 0 and doc[0] == used_category[0]:
            count0 = count0 + 1
        elif labels[i] == 0 and doc[0] == used_category[1]:
            count1 = count1 + 1
    if count0 > count1:
        sum_count = count0
    else:
        sum_count = count1
    # consider cluster 1
    cindex = class_map[used_category[0]]
    count0 = docnum[cindex] - count0 # number of used_category[0] docs in cluster 1
    count1 = docnum[1 - cindex] - count1 # number of used_category[1] docs in cluster 1
    if count0 > count1:
        sum_count = sum_count + count0
    else:
        sum_count = sum_count + count1
    
    purity = sum_count * 1.0 / N
    print "Purity: " + str(purity)

"""
Evaluate classification result
"""
def evaluate_classification(labels, tdocset, N):
    testN = len(tdocset)
    trainN = N - testN
    correct = 0 # number of correctly classified
    for i in range(testN):
        category = tdocset[i][0]
        if category == used_category[0] or category == used_category[1]:
            if labels[i+trainN] == class_map[category]:
                correct = correct + 1
    accuracy = correct * 1.0 / testN
    output_file.write("Classification accuracy: " + str(accuracy) + "\n")
    print "Classification accuracy: " + str(accuracy)
    
# main loop    
def main(argv):
#    print "File to open: " + argv[1]
    if len(argv) != 4:
        print "Usage: python ./NB-with-Gibbs.py data_file test_file max_iters"
        exit()

    print "Loading training data..."
    (docset, docindexset, docnum) = load_data(argv[1])
    
    # initializations for parameters
    T = int(argv[3]) # max iterations
    trainN = len(docset) # number of documents
    print "Loading training data done."
    print "Categories used for classification: " + used_category[0] + " and " + used_category[1]
    print "Total number of training documents:" + str(trainN)
    print "Number of training documents in category '" + used_category[0] + "':" + str(docnum[class_map[used_category[0]]])
    print "Number of training documents in category '" + used_category[1] + "':" + str(docnum[class_map[used_category[1]]])
    
    print "Loading test data..."
    (tdocset, tdocindexset, tdocnum) = load_data(argv[2])
    print "Loading test data done."
    testN = len(tdocset)
    print "Total number of test documents:" + str(testN)
    print "Number of test documents in category '" + used_category[0] + "':" + str(tdocnum[class_map[used_category[0]]])
    print "Number of test documents in category '" + used_category[1] + "':" + str(tdocnum[class_map[used_category[1]]])
    V = len(vocabulary) # size of the vocabulary
    print "Size of vocabulary: " + str(V)
    
    #randomly set labels for documents
    N = trainN + testN
    labels = [0] * N # labes for all documents, 0 or 1
    class_count = [0, 0] # number of documents in each class
    for i in range(trainN): # assign labels for training doccument
        category = docset[i][0]
        labels[i] = class_map[category]
    for i in range(testN): # randomly assign test document with labels 0 or 1
        label = random.randint(0, 1)
        labels[i + trainN] = label
        class_count[label] = class_count[label] + 1
        
    # count word in each class
    word_count = [[0]*V, [0]*V]
    # extend training document list with test document
    docset.extend(tdocset)
    docindexset.extend(tdocindexset)
    count_word(docindexset, labels, word_count)
    
    hyper_gamma = (2, 2) # parameter for Beta distribution
    hyper_multi = [1] * V # hyperparameter vector for the multinominal prior
    # theta0 and theta1 are the multinominal prior for words
    theta0 = Dirichlet(hyper_multi)
    theta1 = Dirichlet(hyper_multi)
    theta = [theta0, theta1]
    
    B = T / 3; # B is for burn-in iterations
    class_vote = [0] * testN # collect vote for every iteration
    print "Start to iterate..."
    for i in range(T):
        change_count = 0
        for j in range(trainN, N): # skip training documents
            label = labels[j]
            assert(label == 0 or label == 1)
            class_count[label] = class_count[label] - 1
            update_word_count(docindexset[j], label, -1, word_count) # update word count by removing the docuemnt j
            
            ln_Pr0 = caculate_Pr(0, class_count, N, j, docindexset, hyper_gamma, theta[0])
            ln_Pr1 = caculate_Pr(1, class_count, N, j, docindexset, hyper_gamma, theta[1])
            # sometimes, the probability is very overwhelming
            if ln_Pr0 - ln_Pr1 > 13.81: # log(999999) \approx 13.815509557963773
                index = 0
            elif ln_Pr1 - ln_Pr0 > 13.81:
                index = 1
            else:
                ratio = exp(ln_Pr0 - ln_Pr1)
                index = choose((0, 1), [ratio, 1]) # choose according a distribution
            new_label = (0, 1)[index]
            
            labels[j] = new_label
            if label != new_label:
                change_count = change_count + 1
            class_count[new_label] = class_count[new_label] + 1
            update_word_count(docindexset[j], new_label, +1, word_count) # update word count by adding the docuemnt j
        
        update_theta(docindexset, labels, theta, hyper_multi, word_count)
        if(i >= B): # start to record data
            for k in range(testN):
                class_vote[k] = class_vote[k] + labels[trainN + k] # record all 1s by adding every label, a bit hacking here
        output_file.write("iteration #" + str(i) + ":\n")
        output_file.write("Number of changes:" + str(change_count) + "\n")
        print "iterationo #:" + str(i)
        #print "Labels:"; print labels
        # evaluate in every iteration
        evaluate_classification(labels, tdocset, N)
    
    # determin labels for each document
    for k in range(testN):
        if class_vote[k] >= T - B - class_vote[k]:
            labels[k] = 1
        else:
            labels[k] = 0
                
    print "Iteration done."
    output_file.write("Results - Labels of testing document:\n")
    output_file.write(str(labels[trainN: N]) + "\n")
    #evaluate_cluster_result(labels, docset, N, docnum)
    evaluate_classification(labels, tdocset, N)
    output_file.close()
    
if __name__ == "__main__":
    main(sys.argv)
