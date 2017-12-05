# -*- coding: utf-8 -*-
"""
Created on Sun Dec 3 12:24:13 2017
@author: marti
"""
import numpy as np
import pandas as pd
import copy
import pickle
from scipy.spatial.distance import cosine

import warnings
#warnings.filterwarnings("ignore")


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
        
def get_neighbors(training, sample):
    
    dist_row = pd.DataFrame({"distance": np.zeros(len(training))}) # create empty dist row
    train_with_dist = copy.copy(pd.concat([training, dist_row], axis = 1)) # append distrow to training data

    for x in range (len(training)): # compute cosine distance for every trainings sample
        d = cosine(sample, training.iloc[x])
        train_with_dist.iloc[x].distance = d
        
    sorted_tuples = train_with_dist.sort_values(by="distance", ascending = True)  # sort list after attribute attr
    # now the k first entrys in sorted tuples are the k nearest neighbors of test-sample
    #print(cosine(sample, sorted_tuples[0])
    return sorted_tuples[:K] # return k first elements as k nearest neighbors

    
def get_prediction(training, sample):
    neighbors = get_neighbors(training, sample)
    #print(neighbors)
    if neighbors.class_label.mean() >= 0.5: # easy implementation of a majority vote
        return 1
    else:
        return 0


def predict_all(training,test):
    correct_sampled = 0
    for sample in range (len(test)):   
        sample_class = get_prediction(training, test.iloc[sample])
        #print(sample_class)
        #print(test.iloc[sample].class_label)
        if sample_class == test.iloc[sample].class_label:
            correct_sampled +=1

    print("Accuracy: " + str(correct_sampled / float(len(test))))
        
def get_csv_plus_header(data):
    header = []
    csv_len = len(pd.read_csv(data).columns)
    for i in range (csv_len-1):
        name = "Attr_" + str(i)
        header.append(name)
    header.append("class_label")
    csv = pd.read_csv(data, names = header)
    return csv
    
#------------------------------------------------------
"""            Hyper-Parameters            """
K = 3
#------------------------------------------------------

# Main
training = get_csv_plus_header('spam_training.csv')
test = get_csv_plus_header('spam_test.csv')

training, test = modify_data(training, test)
predict_all(training, test)


