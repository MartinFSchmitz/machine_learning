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
from Pruning_heuristic import prune_heuristicly
from Pruning_petr import prune_petr
import warnings
warnings.filterwarnings("ignore")

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
    if len(part_ex) == 0:
        return 0
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

    return highest_gain, split_value

def select_split_attribute(atts, node):  # ToDo (only select attributes that were not chosen before)
    
    split_attr = atts[0] #initialize variables
    max_inf_gain = 0
    split_value = 0
    
    for x in atts:
        if not x in node.used_atts:  # if attribute is not already used in upper part of the tree
            
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

    
    
def TDIDT( atts, node, depth, gparent, nparent, graph): # list of attributes, current node, current depth of the tree    
    # Some reasons why we should stop searching...
    
    if depth > 3: # depth of the tree is 3 
        #make leafs under that node
        make_leaf_with_fitting_class(node)
        g_node = graphstuff(node, gparent, nparent, graph)
        return
    depth += 1
    # are the examples perfectly classified?
    all_classified, c = all_examples_classified(node.examples)
    if all_classified:
        node.becomes_leaf(c)
        g_node = graphstuff(node, gparent, nparent, graph)
        return
        
    if (len(atts) <= len(node.used_atts)): # if we already used all attributes, we have a leaf
        make_leaf_with_fitting_class(node)
        gg_node = graphstuff(node, gparent, nparent, graph)
        return
        
        
    #-----------------------------------------
    # Begin Algorithm
    
    
    split_att, split_val = select_split_attribute(atts, node)  # do the computation which attribute to split with
    
    examples_left = node.examples[getattr(node.examples,split_att) <= split_val]  # divide examples, use getattr(obj,attr) to  
    examples_right = node.examples[getattr(node.examples,split_att) > split_val]  # transform variable into attribute
        
    used_atts = copy.copy(node.used_atts)
    used_atts.append(split_att)  # append the currently used attribute to list of used attributes
    """
    print("used_atts: ")
    print(used_atts)
    print("node.used_atts: ")
    print(node.used_atts)
    """
    
    
    # make new nodes for next iteration
    new_node_left = Node(used_atts = used_atts, examples = examples_left)  
    new_node_right = Node(used_atts = used_atts, examples = examples_right)
    node.left = new_node_left
    node.right = new_node_right
    
    node.split_value = split_val
    node.split_attr = split_att
    node.label = split_att + " <= " + str(split_val)  #"s% <= d%" %(split_att, split_val)
    node.label += '\n' + "Samples: " + str(len(node.examples)) + '\n' + "trisomic: " + str(len(node.examples[node.examples.class_label == 1]))
    node.label += " " + "healthy: " +  str(len(node.examples[node.examples.class_label == 0]))
    
    # draw everything into a pydot/ graphviz graph
    g_node = graphstuff(node, gparent, nparent, graph)
    
    # do the algorithm rekursively
    TDIDT(atts, new_node_left, depth, g_node, node, graph)  
    TDIDT(atts, new_node_right, depth, g_node, node, graph)
        

def graphstuff(node, gparent, nparent, graph):      
    global nodenr
    
    if (node.leaf):
        g_node = pydot.Node(nodenr, shape = "oval", label= node.label , style="solid", fillcolor="red")
    else:
        g_node = pydot.Node(nodenr, shape = "box", label= node.label , style="solid", fillcolor="red")
    graph.add_node(g_node)
    if gparent != None:
        if (nparent.left == node):
            graph.add_edge(pydot.Edge(gparent, g_node, label = "True"))
        else:
            graph.add_edge(pydot.Edge(gparent, g_node, label = "False"))

    nodenr += 1    
    return g_node

def intializeTDIDT(dataFrame, graphName):
    global nodenr
    nodenr = 0

    tree = Node(examples = dataFrame) #create an empty root node that will grow to a tree
    tree.used_atts.append('class_label')
    graph = pydot.Dot(graph_type='graph')

    print("")
    print("         Start of TDIDT")
    TDIDT(atts, tree, 0, None, None, graph) # afterwards tree is a real tree
    print("         Finish of TDIDT")

    print("")
    print("         Outputing Tree Data")
    pickle.dump(tree, open(graphName + ".p", "wb"))   

    graph.write_png(graphName + '.png')
    graph.write_dot(graphName + '.dot')

def correct_labeled(x, node):
    
    if(node.leaf):
        if(x.class_label == node.label):
            return 1
        else:
            return 0
    else:
    
        value = getattr(x, node.split_attr)
        
        if ( value <= node.split_value):
            return correct_labeled(x, node.left)
            
        else:
            return correct_labeled(x, node.right)


def flipAttributes(dataFrame, percentage):
    dataFrameCopy = copy.copy(dataFrame)

    iter_csv = dataFrameCopy.itertuples() # only itertuples brings the fitting format
    last = next(iter_csv) # skip first entry
    i = 0
    j = 0
    for x in iter_csv:
        rnd = np.random.random()
        if (rnd < percentage):
            j += 1
            if (x.class_label == 0):
                dataFrameCopy.set_value(i, 'class_label', 1)
            else:
                dataFrameCopy.set_value(i, 'class_label', 0)
        i += 1
    print("         Labels Flipped: " + str(j) + " / " + str(len(dataFrameCopy)))
    print("         Percentage: " + str(j / len(dataFrameCopy)))    

    return dataFrameCopy

def calculateAccuracy(graphName):
    csv = pd.read_csv('gene_expression_test.csv')
    tree = pickle.load( open( graphName + ".p", "rb" ) )
    correct = 0

    iter_csv = csv.itertuples() #brings fitting format of one example
    
    for x in iter_csv:
        correct += correct_labeled(x, tree)

    accuracy = float(correct) / float(len(csv))
         
    print(graphName + " Correct: " + str(correct) + " Sample Size: " + str(len(csv)) + " Accuracy: " + str(accuracy))  


nodenr = 0   
csv = pd.read_csv('gene_expression_training.csv')
atts = csv.columns  # if you want to delete one entry: .delete(0)
used_atts = []  # list with used attributes
                # use: used_atts.__contains__(x) to ask if indices there
                # and used_atts.append(x) to append index

print("")
print("Start of Excersise Sheet #2 calculations")
print("     Start of Task 4: ")
intializeTDIDT(csv,     "Excersise#4")
print("     Finish of Task 4")

"""
print("")
print("     Start of Task 5.3: ")
print("         Start of Task 5.3a:")
csv01 = flipAttributes(csv, 0.1)
intializeTDIDT(csv01, "Excersise#5_3a")
print("         Finish of Task 5.3a")

print("")
print("         Start of Task 5.3b:")
csv025 = flipAttributes(csv, 0.25)
intializeTDIDT(csv025, "Excersise#5_3b")
print("         Finish of Task 5.3b")
print("     Finish of 5.3")
print("Finish of Excersise Sheet #2 calculations")
print("")

print("Start of Excersise Sheet #3 calculations")

print("     Start of pruning heuristic")
print("")
prune_heuristicly("Excersise#4", "Excersise#4_pruned_heuristic", csv)
prune_heuristicly("Excersise#5_3a", "Excersise#5_3a_pruned_heuristic", csv01)
prune_heuristicly("Excersise#5_3b", "Excersise#5_3b_pruned_heuristic", csv025)
print("")
print("     Finish of pruning heuristic")
print("")
"""
print("     Start of pruning petr")
print("")
prune_petr("Excersise#4", "Excersise#4_pruned_petr", csv)
#prune_petr("Excersise#5_3a", "Excersise#5_3a_pruned_petr", csv01)
#prune_petr("Excersise#5_3b", "Excersise#5_3b_pruned_petr", csv025)
print("")
print("     Finish of pruning petr")
print("")
print("")

"""
print("     Start of Classification Rules")
print("")

TODO

print("")
print("     Finish of Classification Rules")
print("")
print("")
"""
print("Finish of Excersise Sheet #3 calculations")
print("")
print("")

print("Accuracy of unpruned trees:")
calculateAccuracy("Excersise#4")
#calculateAccuracy("Excersise#5_3a")
#calculateAccuracy("Excersise#5_3b")
print("")
#print("Accuracy of heuristicly pruned trees:")
#calculateAccuracy("Excersise#4_pruned_heuristic")
#calculateAccuracy("Excersise#5_3a_pruned_heuristic")
#calculateAccuracy("Excersise#5_3b_pruned_heuristic")
print("")
print("Accuracy of petr pruned trees:")
calculateAccuracy("Excersise#4_pruned_petr")
#calculateAccuracy("Excersise#5_3a_pruned_petr")
#calculateAccuracy("Excersise#5_3b_pruned_petr")
print("")
print("Accuracy of Classification Rules:")

