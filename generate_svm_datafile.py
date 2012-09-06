#! /usr/bin/env python
import sys
import math

from operator import itemgetter # for sort

from naive_bayes import build_vocabulary, extract_token

"""
Generate data files for SVM training
Detail:
Use tf*idf as features for words
"""

NUM_CATEGORY = 2 # number of categories
magic_key_set = set(["_CATEGORY", "_NUM_WORDS"])

# caculate idf for each term in vocabulary
def caculate_idf(traindoclist, vocabulary):
    N = len(traindoclist)
    for key, value in vocabulary.iteritems():
        if key in magic_key_set:
            continue
        count = 1 # just in case count == 0 after the searching below
        for doc in traindoclist:
            if key in doc:
                count = count + 1
        idf = math.log(N * 1.0 / count)
        value = [value[0], value[1], idf]
        vocabulary[key] = value

def load_data(filepath, vocabulary):
    fsource = open(filepath, "r")
    doclist = []
    for line in fsource:
        (docdict, category) = extract_token(line)
        if category == -1:
            continue
        build_vocabulary(docdict, vocabulary)
        doclist.append(docdict)
        
    print "Number of docs:  " + str(len(doclist))
    
    return doclist
    
# generate svm-format data file
# Use tf*idf as features, and category "course" --> -1, "faculty" --> 1
def generate_data_file(doclist, outfile_path, vocabulary):
    print "Writting to file: " + outfile_path
    outfile = open(outfile_path, "w")
    for doc in doclist:
        features = [] # element will be: [index, tf*idf]
        if doc["_CATEGORY"] == 0:
            label = -1
        else:
            label = +1
        outfile.write(str(label) + " ") # output the label
        for key, value in doc.iteritems():
            if key in magic_key_set:
                continue
            idf = vocabulary[key][2]
            index = vocabulary[key][0] + 1 # feature index must start with 1, not 0
            tf = doc[key][1]
            features.append([index, tf*idf])
        features = sorted(features, key=itemgetter(0)) # sort by index
        for f in features:
            outfile.write(str(f[0]) + ":" + str(f[1]) + " ")
        outfile.write("\n")
    outfile.close()

def main(argv):
    print "Usage: python ./generate-svm-datafile train_file test_file"
    vocabulary = dict() # vocabulary for the whole dataset
    traindoclist = load_data(argv[1], vocabulary)
    testdoclist = load_data(argv[2], vocabulary)
    print "Vocabulary size: " + str(len(vocabulary))
    caculate_idf(traindoclist, vocabulary)
    # output the data file
    generate_data_file(traindoclist, argv[1] + "-svm-data.txt", vocabulary)
    generate_data_file(testdoclist, argv[2] + "-svm-data.txt", vocabulary)
    
if __name__ == "__main__":
    main(sys.argv)
