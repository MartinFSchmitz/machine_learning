# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:24:13 2017

@author: marti
"""
import numpy as np
import pandas as pd
import copy
import pydot
import os
import pickle
from Node import Node

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def all_examples_classified(examples):
    # is the data the same, when i remove all data classified with 1 or 0? 
    a = len(examples[examples.class_label == 1]) is len(examples)
    b = len(examples[examples.class_label == 0]) is len(examples)
    # in which class is it classified?
    
    if a>b:
        c = 1
    else: c = 0
    return a or b, c
    
def entropy (a,b):
    if a == 0 or b == 0:
        return 0
    else:
        return a * np.log2(1/a) + b * np.log2(1/b)

def conditional_entropy (all_ex,part_ex):
    pos = part_ex[part_ex.class_label == 1]  # all examples in part_ex that are labeled with 1 / 0
    #neg = part_ex[part_ex.class_label == 0]

    frac = len(part_ex)/len(all_ex)  # amount of sub examples in this part through total examples in node
    p_l = len(pos)/ len(part_ex)  # compute p_i's
    p_r = 1-p_l  # len(neg)/ len(part_ex)
    h = entropy(p_l,p_r)
    return frac * h   # H(S|A)
                        
                        
def  compute_gain(attr, split_val, node):
    # Gain(A,S) = H(S) - H(S|A)
    attr_ex = getattr(node.examples,attr)  # all examples with their attr value
    examples_left = node.examples[attr_ex <= split_val]  # examples left from split value
    examples_right = node.examples[attr_ex > split_val]  # " " right from split value
    
    p_l = len(examples_left)/len(attr_ex)  # probability for being left to the split value
    p_r = 1-p_l # prob of being right to the split vaule
    h_s = entropy(p_l, p_r)
    h_s_l = conditional_entropy(attr_ex,examples_left)
    h_s_r = conditional_entropy(attr_ex,examples_right)
    
    return h_s - (h_s_l + h_s_r)   # Gain(S|A)

def compute_split_value(attr, node): # compute highest possible information gain for one specific attribute, trying different cutting values
    # for continues attribut space in the decision tree
    highest_gain = 0
    split_value = 0
    
    sorted_csv = node.examples.sort_values(by=attr)  # sort list after attribute attr
    iter_csv = sorted_csv.itertuples() # only itertuples brings the fitting format
    last = next(iter_csv) # skip first entry

    
    for x in iter_csv:
        if(x.class_label != last.class_label):
            mean = (getattr(x,attr) + getattr(last,attr))/2 #compute mean as posible cutting value
            gain = compute_gain(attr, mean, node)
            if gain > highest_gain:
                highest_gain = gain
                split_value = mean
        last = x

    return 1,0.3

def select_split_attribute(atts, node):  # ToDo (only select attributes that were not chosen before)

    split_attr = atts[0] #initialize variables
    max_inf_gain = 0
    split_value = 0
    
    for x in atts:
        if not node.used_atts.__contains__(x):  # if attribute is not already used in upper part of the tree
            
            inf_gain, split = compute_split_value(x, node)  # compute highest possible gain for attribute x and the according split value
            
            if inf_gain > max_inf_gain:  # is attribute better than earlier ones?
                max_inf_gain = inf_gain
                split_attr = x
                split_value = split
    #print(split_attr)
    return split_attr, split_value

    
def make_leaf_with_fitting_class(node):
    if ( len(node.examples[node.examples.class_label == 1]) > len(node.examples[node.examples.class_label == 0])):
        c = 1
    else: c = 0
    node.becomes_leaf(c)

    
    
def TDIDT( atts, node, depth): # list of attributes, current node, current depth of the tree
    #node_list.append(node)
    
    # Some reasons why we should stop searching...
    
    if depth > 3: # depth of the tree is 3 
        #make leafs under that node
        make_leaf_with_fitting_class(node)
        return
    depth += 1
    # are the examples perfectly classified?
    all_classified, c = all_examples_classified(node.examples)
    if all_classified:
        node.becomes_leaf(c)
        return
        
    if (len(atts) <= len(node.used_atts)): # if we already used all attributes, we have a leaf
        make_leaf_with_fitting_class(node)
        return
        
        
    #-----------------------------------------
    # Begin Algorithm
    
    
    split_att, split_val = select_split_attribute(atts, node)  # do the computation which attribute to split with
    
    examples_left = node.examples[getattr(node.examples,split_att) <= split_val]  # divide examples, use getattr(obj,attr) to  
    examples_right = node.examples[getattr(node.examples,split_att) > split_val]  # transform variable into attribute
        
    used_atts = copy.copy(node.used_atts)
    used_atts.append(split_att)  # append the currently used attribute to list of used attributes
    #print(used_atts)
    
    
    # make new nodes for next iteration
    new_node_left = Node(used_atts = used_atts, examples = examples_left)  
    new_node_right = Node(used_atts = used_atts, examples = examples_right)
    node.left = new_node_left
    node.right = new_node_right
    
    node.split_value = split_val
    node.split_attr = split_att
    ###node.label = "ha"
    """
    # draw everything into a pydot/ graphviz graph
    label =  "ha" #split_att + " <= " + str(split_val)  #"s% <= d%" %(split_att, split_val)
    g_node = pydot.Node("node", shape = "box", label= label , style="solid", fillcolor="red")
    graph.add_node(g_node)
    if parent != None:
        graph.add_edge(pydot.Edge(parent, g_node))
    """
    # do the algorithm rekursively
    TDIDT(atts, new_node_left, depth)  
    TDIDT(atts, new_node_right, depth)
        
        
    
csv = pd.read_csv('gene_expression_training.csv')
atts = csv.columns  # if you want to delete one entry: .delete(0)
used_atts = []  # list with used attributes
                # use: used_atts.__contains__(x) to ask if indices there
                # and used_atts.append(x) to append index

tree = Node(examples = csv) #create an empty root node that will grow to a tree
#graph = pydot.Dot(graph_type='graph')
node_list = []
print("start")
TDIDT(atts, tree, 0) # afterwards tree is a real tree
print("finished")

pickle.dump(tree, open("tree.p", "wb"))

"""
g = pydot.Node("node", shape = "box", label = "tree.label" , style="solid", fillcolor="red")
iter_nodes = iter(node_list)
next(iter_nodes)
for n in node_list:
"""
    

#graph.write_png('example1_graph.png')



# csv.values[x] gives values of line x, csv[0:2] gives the first two lines + header
# csv.columns gives the name of the columns, so the header line
# csv.iloc[3:5,0:2] gives the data in typed range
# sorted_csv = csv.sort_values(by='class_label')
# csv[csv.class_label == 1] split by dataaa value !!!!!!!!
# sorted_csv.split
# print(csv)


