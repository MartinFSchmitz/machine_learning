# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:24:13 2017

@author: marti
"""

class Node(object):
    def __init__(self, used_atts = [], examples = None, leaf = False, label = None):
        
        self.used_atts = used_atts
        self.examples = examples
        
        self.left = None
        self.right = None
        
        self.split_attr = None
        self.split_value = None
        self.label = label
        self.leaf = leaf

        
    def becomes_leaf(self,c):
        self.leaf = True
        self.label = c
        
