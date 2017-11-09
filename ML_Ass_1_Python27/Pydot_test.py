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
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



# first you create a new graph, you do that with pydot.Dot()
graph = pydot.Dot(graph_type='digraph')
label= "Hallo = %d %d" % (5,4)
node_a = pydot.Node(shape = "box", label= label , style="solid", fillcolor="red")
graph.add_node(node_a)


node_b = pydot.Node(shape = "box", style="solid", fillcolor="red")
graph.add_node(node_a)
node_c = pydot.Node(shape = "box", style="solid", fillcolor="red")
graph.add_node(node_a)

graph.add_edge(pydot.Edge(node_a, node_b))

graph.add_edge(pydot.Edge(node_a, node_c))
graph.add_edge(pydot.Edge(node_a, node_a, label="and back we go again", labelfontcolor="#009933", fontsize="10.0", color="blue"))

graph.write_png('example1_graph.png')

# and we are done!