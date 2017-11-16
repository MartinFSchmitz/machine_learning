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
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def correct_labeled(x, node, current_node, passed_node = False) :  # gives -1 if example doesnt pass current_node,
                                
    if(node == current_node):  # and 1/0 for the respective class the example fits to
        passed_node = True # If it passed the node at some point, the example will be added later
    if(node.leaf):
        if(passed_node == False):
            return -1
        if(x.class_label == node.label):
            return 1
        else:
            return 0
    else:
        
    
        value = getattr(x, node.split_attr)
        if ( value <= node.split_value):
            return correct_labeled(x, node.left, current_node, passed_node)           
        else:
            return correct_labeled(x, node.right, current_node, passed_node)
        
        
def compute_acc(correct, total_len): # compute accuracy from respective node
    if total_len == 0: accuracy = 1
    else: accuracy = correct / total_len
    return accuracy
        
def compute_prun_acc(labels, total_len): # computes acuracy when node is a leafnode and fitting class
    if total_len == 0:
        prun_accuracy = 1
    else:
        prun_accuracy = labels / total_len 
        
    if prun_accuracy < 0.5:  # prun accuracy counts the labels of 1 or 0
        prun_accuracy = 1 - prun_accuracy
        prun_class = 0 # class after pruning
    else: prun_class = 1
    return prun_accuracy, prun_class
                
def compute_accuracys(current_node):
    iter_csv = CSV.itertuples() #brings fitting format of one example
    correct = 0.
    labels = 0.
    prun_class = 1 
    
    total_len = float(len(CSV)) # total length of all examples

    for x in iter_csv:
        i= correct_labeled(x, TREE, current_node) # i says if the label was correct
        if i == -1:  # -1 means x didnt pass node
            total_len -= 1  # than the total size should shrink by one because this sample is not viewed
        else:
            labels += x.class_label
            correct += i
            
    acc = compute_acc(correct, total_len)
    prun_acc, prun_class = compute_prun_acc(labels, total_len)

    return acc, prun_acc, prun_class                       

def make_node_to_leaf(node, prun_class):
    node.left = None
    node.right = None
    node.leaf = True
    node.label = prun_class
    
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

def pruning(node):
    accuracy, prun_accuracy , prun_class = compute_accuracys(node)
    #print(accuracy, prun_accuracy)
    # ToDo make new node and substitude old one
    if prun_accuracy >= accuracy:
        make_node_to_leaf(node, prun_class)
        print("pruned")
        
def post_order_traversal(node):
    if(node.leaf): return
    post_order_traversal(node.left)
    post_order_traversal(node.right)
    pruning(node)
    
def graphstuff(node, gparent, nparent):      
    global nodenr
    
    if (node.leaf):
        g_node = pydot.Node(nodenr, shape = "oval", label= node.label , style="solid", fillcolor="red")
    else:
        g_node = pydot.Node(nodenr, shape = "box", label= node.label , style="solid", fillcolor="red")
    GRAPH.add_node(g_node)
    if gparent != None:
        if (nparent.left == node):
            GRAPH.add_edge(pydot.Edge(gparent, g_node, label = "True"))
        else:
            GRAPH.add_edge(pydot.Edge(gparent, g_node, label = "False"))

    nodenr += 1    
    return g_node
    
def pre_order_traversal(node, g_parent, n_parent):
    if node == None: return
    g_node = graphstuff(node, g_parent, n_parent)
    pre_order_traversal(node.left, g_node, node)
    pre_order_traversal(node.right, g_node, node)


def prune_heuristicly(graphNameToPrune, newGraphName, csv):
    global CSV
    global TREE
    global nodenr
    global GRAPH 

    nodenr = 0
    GRAPH = pydot.Dot(graph_type='graph')

  
    CSV = csv
    TREE = pickle.load( open(graphNameToPrune + ".p", "rb") )
    post_order_traversal(TREE)
    pre_order_traversal(TREE, None, None)


    pickle.dump(TREE, open(newGraphName + ".p", "wb"))
    GRAPH.write_png(newGraphName + '.png')
    GRAPH.write_dot(newGraphName + '.dot')

CSV = 0
TREE = 0
nodenr = 0
GRAPH = 0
          