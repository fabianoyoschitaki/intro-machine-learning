#!/usr/bin/python

""" 
    Calculating Entropy in a dataset of 2 classes 
	evenly split, which yields entropy = 1, the max impurity situation
"""
    
import math
#Pslow 2/4 and Pfast 2/4

pslow = 2.0/4
pfast = 2.0/4
entropy = -(pslow) * math.log(pslow, 2) -(pfast) * math.log(pfast, 2)
print "entropy: ", entropy