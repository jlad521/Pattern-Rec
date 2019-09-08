'''
Justin Lad & Richard (Corey) Riehm
Handwriting Classifier
4/2/2019
'''

import os
import glob
#import xml.etree.ElementTree as ET 
import csv
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
import pickle


#reads a GT file and returns a dictionary with UI as key and the symbol as value
#serves as a lookup
def read_GT_file(f_name):
    GT_dict = {}
    f = open(f_name,'r')
    text_list = f.readlines()
    for line in text_list:
        splt = line.split(',')
        GT_dict[splt[0]] = splt[1]
    return GT_dict    
    
    
    

#splits the data into a 70/30 training/testing split while maintaining balanced within each class    
#train_dict is a dictionary with symbols as the keys and the UI's associated with them in a list as the values
#returns a dictionary for training and a dictionary for testing
def split_data(train_dict):   
    test_dict = {}
    random.seed(1234)
    #iterate over each symbol
    for lab in train_dict:
        #get length of the list of UI's to be used to removed 30%
        length = len(train_dict[lab])
        #initialize the key to an empty list
        test_dict[lab] = []
        for i in range(math.ceil(length * .3)):
            #removes random UI from the list and adds to the new dictionary
            random_index = random.randint(0,len(train_dict[lab])-1)
            test_dict[lab].append(train_dict[lab][random_index])
            del(train_dict[lab][random_index])

    return train_dict, test_dict



def read_inkml():
    #read_truth_file() # use global var
    
    truth_dict = {}
    junk_dict = {}
    truth_GT_dict = read_GT_file('iso_GT.txt')    #maybe should unpickle? 
    junk_GT_dict = read_GT_file('junk_GT_v3.txt')    #maybe should unpickle?
    
    
    
    for key in truth_GT_dict:
        junk_GT_dict[key] = truth_GT_dict[key]     
    

    truth_dct_GT = {}
    
    for filename in glob.glob('../task2-trainSymb2014(1)/trainingSymbols/*.inkml'):   
        
        string_label,UI = parse_soup(filename, truth_GT_dict)
        
        
        if string_label in truth_dct_GT:
            truth_dct_GT[string_label].append(UI)
        else:
            truth_dct_GT[string_label] = [UI]
    
    
        if string_label in truth_dict:
            truth_dict[string_label].append(filename)
        else:
            truth_dict[string_label] = [filename]            
    
    true_train,true_test = split_data(truth_dict)
    
    truth_pkl = open('truth_GT.pkl', 'wb')
    pickle.dump(truth_dict,truth_pkl)
    
    write_to_csv(true_test, "true_test.csv")
    write_to_csv(true_train, "true_train.csv")
    
    
    
    
    true_train,true_test = split_data(truth_dct_GT)
    
    
    
    #filenames
    write_GT_csv(true_test, 'true')
    

    junk_dct_GT = {}
        
     
    for filename in glob.glob('../task2-trainSymb2014(1)/trainingJunk/*.inkml'):
        string_label, UI = parse_soup(filename, junk_GT_dict)
        
        
        if string_label in junk_dct_GT:
            
            junk_dct_GT[string_label].append(UI)
        else:
            junk_dct_GT[string_label] = [UI]        
            
            
        if string_label in junk_dict:
            junk_dict[string_label].append(filename)
        else:
            junk_dict[string_label] = [filename]
            
    for key in truth_dict:
        junk_dict[key] = truth_dict[key]    
    
    for key in truth_dct_GT:
        junk_dct_GT[key] = truth_dct_GT[key]     
    
    junk_train, junk_test = split_data(junk_dict)
    
    write_to_csv(junk_test, "both_test.csv")
    write_to_csv(junk_train, "both_train.csv")
    
    
    junk_pkl = open('both_GT.pkl', 'wb')
    pickle.dump(junk_dict,junk_pkl)   
    
    junk_train, junk_test = split_data(junk_dct_GT)
    

    write_GT_csv(junk_test, 'both')
    

#writes the GT csv file, where each row is UI, lab
def write_GT_csv(dct, val):
    csv_writer = csv.writer(open(val + '_test_GT.csv','w',newline='\n'))
    for lab in dct:
        for UI in dct[lab]:
            csv_writer.writerow([UI, lab])
    
#writes the input csv file containing filenames
def write_to_csv(dct, fname):
    csv_writer = csv.writer(open(fname,'w',newline='\n'))
    for UI in dct:
        for filename in dct[UI]:
            csv_writer.writerow([filename])
        

def parse_soup(file, GT_dict):
    with open(file) as f:
        soup = BeautifulSoup(f)
        
    for b in soup.findAll('annotation'):
        if b['type'] == 'UI':
            UI = b.contents[0].strip('\n')
            try:
                string_label = GT_dict[UI].strip('\n')
            except:
                string_label = "invalid"
            if '\\' in string_label:
                string_label = string_label[1:]
    return string_label,UI
            
def main():
    read_inkml()
    
main()