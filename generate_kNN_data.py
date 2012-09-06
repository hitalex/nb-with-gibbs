#! /usr/bin/env python
import sys
from operator import itemgetter # for sort

from generate_svm_datafile import caculate_idf, load_data

"""
Generate data files for R kNN
"""

magic_key_set = set(["_CATEGORY", "_NUM_WORDS"])

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
        for key, value in vocabulary.iteritems():
            if key in magic_key_set:
                continue
            idf = vocabulary[key][2]
            index = vocabulary[key][0] + 1 # feature index must start with 1, not 0
            if key in doc:
                tf = doc[key][1]
            else:
                tf = 0
            features.append([index, tf*idf])
        features = sorted(features, key=itemgetter(0)) # sort by index
        for f in features:
            outfile.write(str(f[1]) + " ")
        outfile.write("\n")
    outfile.close()

def main(argv):
    print "Usage: python ./generate_kNN_datafile train_file test_file"
    vocabulary = dict() # vocabulary for the whole dataset
    
    traindoclist = load_data(argv[1], vocabulary)
    testdoclist = load_data(argv[2], vocabulary)
    print "Vocabulary size: " + str(len(vocabulary))
    caculate_idf(traindoclist, vocabulary)
    # generate csv format data file
    generate_data_file(traindoclist, argv[1] + "-R-data.csv", vocabulary)
    generate_data_file(testdoclist, argv[2] + "-R-data.csv", vocabulary)    

if __name__ == "__main__":
    main(sys.argv)
