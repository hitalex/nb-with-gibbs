#! /usr/bin/env python
import sys
import math

from naive-bayes import build_vocabulary, extract_token

"""
Generate data files for SVM training
Detail:
Use tf*idf as features for words
"""

NUM_CATEGORY = 2 # number of categories
category_map = dict({"course":0, "faculty":1})

magic_key_set = set(["_CATEGORY", "_NUM_WORDS"])

def main(argv):
    print "Usage: python ./generate-svm-datafile train_file test_file"
    
    
if __name__ == "__main__":
    main(argv)
