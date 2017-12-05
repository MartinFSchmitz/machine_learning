# -*- coding: utf-8 -*-
"""
Created on Sun Dec 3 12:24:13 2017
@author: marti
"""
import numpy as np
import pandas as pd

# file to save every relevant pandas method to reuse easily

csv = pd.read_csv('gene_expression_training.csv')
atts = csv.columns  # if you want to delete one entry: .delete(0)
iter_csv = csv.itertuples() #brings fitting format of one example
#for x in iter_csv:
csv.sort_values(by=attr)  # sort list after attribute attr
csv[attr <= split_val]  # examples left from split value, also with == 0
csv[attr].max() # get maximal number of that attribute, use for normalization
cs.mean(), csv.std() # give csv with means or std_derivations
csv.loc[1] # get first (or second?) example

csv = csv.sort_values(by="attr", ascending = False)  # sort list after attribute attr
sorted_tuples[:K] # take the first k tuples