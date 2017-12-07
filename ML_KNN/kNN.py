# -*- coding: utf-8 -*-
"""
Created on Sun Dec 3 12:24:13 2017
@author: marti
"""
import numpy as np
import pandas as pd
import copy
import pickle
from scipy.spatial import distance
import sys
import time

"""            Hyper-Parameters            """
K = 3
GROUP_DEPENDANT_TASK = False

#------------------------------------------------------

def normalize(train, test):
    """ normalize data sets: search for highest entry for attributes
     and divides attribute values through found max_value """
    words = train.columns 
    for attr in words:
        if attr != "class_label":
            break
            train[attr] = train[attr] / train[attr].max() # divide through maximal
            test[attr] = test[attr] / train[attr].max() # we divide through train_max because normalization has to be same
    return train, test

def compute_new_values(csv, means, std_der):
    """ computing new values in data sets according to formula v_A' = (v_A − m_A)/s_A """

    csv_new = copy.copy(csv)
    words = csv.columns 
    for attr in words: 
        if attr != "class_label":   
            csv_new[attr] = (csv[attr] - means[attr]) / float(std_der[attr])
        
    return csv_new

def modify_data(training, test):
    """ modify training and test data following given algorithm """ 
    training, test = normalize(training, test)
    means = training.mean()
    std_der = training.std() 
    training = compute_new_values(training, means, std_der)
    test = compute_new_values(test, means, std_der)    
    return training, test
        
def get_neighbors(training, sample, k):
    
    dist_row = pd.DataFrame({"distance": np.zeros(len(training))}) # create empty dist row
    train_with_dist = copy.copy(pd.concat([training, dist_row], axis = 1)) # append distrow to training data

    for x in range (len(training)): # compute cosine distance for every trainings sample
        if GROUP_DEPENDANT_TASK:
            d = distance.euclidean(sample, training.iloc[x])
        else:
            d = distance.cosine(sample, training.iloc[x])
        train_with_dist.iloc[x].distance = d
    

    sorted_tuples = train_with_dist.sort_values(by="distance", ascending = True)  # sort list after attribute attr
    # now the k first entrys in sorted tuples are the k nearest neighbors of test-sample
    return sorted_tuples[:k] # return k first elements as k nearest neighbors

    
def get_prediction(training, sample, k):
    neighbors = get_neighbors(training, sample, k)
    if neighbors.class_label.mean() >= 0.5: # easy implementation of a majority vote
        return 1
    else:
        return 0


def compute_accuracy(training,test, k):
    correct_sampled = 0
    for sample in range (len(test)):  
        #print(str(round((sample/float(len(test)))*100,2)) + "%  done")
        sample_class = get_prediction(training, test.iloc[sample], k)
        if sample_class == test.iloc[sample].class_label:
            
            correct_sampled +=1
    print("K = " + str(k))
    print("Accuracy: " + str(correct_sampled / float(len(test))))
        
def get_csv_plus_header(data):
    """ read data and add header line """
    header = []
    csv_len = len(pd.read_csv(data).columns)
    for i in range (csv_len-1):
        name = "Attr_" + str(i)
        header.append(name)
    header.append("class_label")
    csv = pd.read_csv(data, names = header)
    return csv


#------------------------------------------------------

# Main
training = get_csv_plus_header('spam_training.csv')
test = get_csv_plus_header('spam_test.csv')
training, test = modify_data(training, test)
print("start computation with K = 1")
compute_accuracy(training, test, 1)
print("start computation with K = 3")
compute_accuracy(training, test, 3)
print("start computation with K = 5")
compute_accuracy(training, test, 5)
