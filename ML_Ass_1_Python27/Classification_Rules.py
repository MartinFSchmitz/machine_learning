# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:24:13 2017

@author: marti
"""
import numpy as np
import pandas as pd
import pickle
from Node import Node
import copy
import pydotplus as pydot
import os
import estimate_error

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class Rule(object): # rule object containing split_parameters and a method to use the rule
    def __init__(self,split_atts, split_vals, split_directions, class_label):
        self.split_atts = split_atts
        self.split_vals = split_vals
        self.split_dirs = split_directions
        self.class_label = class_label
        
        if (len(split_atts) != len(split_vals) or len(split_vals) != len(split_directions)):
            print("Error: Parameters for this Rule a wrong, size not fitting!")
        
    def use_rule(self, x): # gives class_label back when x fullfills the rule, returns -1 otherwise
        for i in range (0,len(self.split_atts)):
            if  ((self.split_dirs[i] == 'l' and (getattr(x,self.split_atts[i]) > self.split_vals[i]))  # if it doesnt fullfill 
            or (self.split_dirs[i] == 'r' and (getattr(x,self.split_atts[i]) <= self.split_vals[i]))): # this part of the rule, return false
                return -1 # -1 means doesnt fullfill the rule
        return self.class_label


    def print_rule(self):
        rule_string = ""
        for i in range (0,len(self.split_atts)):
            if rule_string != "": rule_string = rule_string + "  and  "
            if self.split_dirs[i] == 'l': sign = "<="   
            else: sign = ">"               
            konj = str(self.split_atts[i]) + " " + sign + " " + str(self.split_vals[i])
            rule_string =  rule_string + konj
        rule_string = rule_string + "  --> " + str(self.class_label)
        return rule_string

def post_order_traversal(node,split_atts, split_vals, split_dirs, split_dir): # travers the tree and creates rules
    dirs = copy.copy(split_dirs)
    if split_dir != None:
        dirs.append(split_dir) # append dir to know if the value of the examples should be less or higher then the split value
        
    if(node.leaf): # create a rule for every leaf
        RULE_SET.append(Rule(split_atts, split_vals, dirs, node.label))
        return
    atts = copy.copy(split_atts) # important to not get side effects
    vals = copy.copy(split_vals)
    atts.append(node.split_attr)
    vals.append(node.split_value)
    post_order_traversal(node.left, atts, vals, dirs, 'l')
    post_order_traversal(node.right, atts, vals, dirs, 'r')

    
def get_error_for_rule(rule):
    
    correct = 0
    total_len= len(CSV)
    iter_csv = CSV.itertuples() #brings fitting format of one example
    for x in iter_csv:     
        rule_label = rule.use_rule(x) # i says if the label was correct   
        if rule_label == -1:  # -1 means x didnt pass node
            total_len -= 1  # than the total size should shrink by one because this sample is not viewed
        else:
            if x.class_label == rule_label: # if rule matches the real class value, it was correct 
                correct += 1   
    y = total_len-correct # true error Y
    error = estimate_error.pessimistic_true_error(y , total_len)
    return error
    
def remove_konjunction_from_rule(rule,x): # cut the konjunction with index x from the rule
    new_split_atts = copy.copy(rule.split_atts)
    new_split_vals = copy.copy(rule.split_vals)
    new_split_dirs = copy.copy(rule.split_dirs)
    del new_split_atts[x]
    del new_split_vals[x]
    del new_split_dirs[x]
    
    return Rule(new_split_atts, new_split_vals, new_split_dirs, rule.class_label)
    
def prun_ruleset():

    for rule in RULE_SET: # go over every rule in ruleset        
        error = get_error_for_rule(rule)
        pruned = True
        while pruned:  # if we pruned it we have to go here again, to remove more konjunctions
            pruned = False  # set true after pruning
            
            for k in range (0, len(rule.split_atts)-1):
                new_rule = remove_konjunction_from_rule(rule,k)
                new_error =  get_error_for_rule(new_rule)
                
                if new_error <= error:
                    error = new_error
                    rule = new_rule
                    pruned = True
                    break
        PRUNED_RULE_SET.append(rule)
CSV = pd.read_csv('gene_expression_test.csv')
TREE = pickle.load( open( "Excersise#5_3a.p", "rb") )
RULE_SET = []
PRUNED_RULE_SET = []
split_attributes = []
split_values = []
split_directions = []
post_order_traversal(TREE, split_attributes, split_values, split_directions, None)  # create unpruned ruleset in RULE_SET

for rule in RULE_SET:
    print(rule.print_rule())
print("------------------------------------------------------------------------")
prun_ruleset()
for rule in PRUNED_RULE_SET:
    print(rule.print_rule())






          