# -*- coding: utf-8 -*-

#! /usr/bin/env python
from random import gammavariate

"""
Samples from a Dirichlet distribution with parameter @alpha using a Gamma distribution
Reference: 
http://en.wikipedia.org/wiki/Dirichlet_distribution
http://stackoverflow.com/questions/3028571/non-uniform-distributed-random-array
"""
def Dirichlet(alpha):
    sample = [gammavariate(a,1) for a in alpha]
    sample = [v/sum(sample) for v in sample]
    return sample
    
if __name__ == "__main__":
    # This is a test
    print Dirichlet([1,1,1]);
