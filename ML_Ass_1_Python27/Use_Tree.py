# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:24:13 2017

@author: marti
"""
import numpy as np
import pandas as pd
import pickle
from Node import Node

def correct_labeled(x, node):
    
    if(node.leaf):
        if(x.class_label == node.label):
            return 1
        else:
            return 0
    else:
    
        value = getattr(x, node.split_attr)
        
        if ( value <= tree.split_value):
            return correct_labeled(x, node.left)
            
        else:
            return correct_labeled(x, node.right)
        
        

        
csv = pd.read_csv('gene_expression_test.csv')
tree = pickle.load( open( "tree.p", "rb" ) )
correct = 0

iter_csv = csv.itertuples() #brings fitting format of one example
    
for x in iter_csv:
    correct += correct_labeled(x, tree)



accuracy = float(correct) / float(len(csv))
         
print(correct, len(csv), accuracy)                       