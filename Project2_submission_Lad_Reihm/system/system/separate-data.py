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
import copy
from joblib import dump,load
import shutil
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
    
    

def split_data_KL(symbol_to_file_dict, file_to_symbol_dict):
    test_dict_files = {}

    train_dict_symbols = {}
    test_dict_symbols = {}
    sum_train = 0
    sum_test = 0
    '''
    for file in train_dict_files:
        sum_train += len(train_dict_files[file])
        for lab in train_dict_files[file]:
            if lab in train_dict_symbols:
                train_dict_symbols[lab] += 1
            else:
                train_dict_symbols[lab] = 1

'''
    test_file_to_symbol_dict = {}
    test_symbol_to_file_dict = {}
    length = sum_train
    length_train = len(file_to_symbol_dict)
    c_length_test = 0
    #for i in range(math.ceil(length * .3)):
    while len(test_file_to_symbol_dict) < (length_train * .3 ):
        random_file = random.choice(list(file_to_symbol_dict))
        symbol_list = file_to_symbol_dict[random_file]
        if len(symbol_to_file_dict) != len(test_symbol_to_file_dict):
            # removes random UI from the list and adds to the new dictionary
            for lab in symbol_list:
                if lab not in test_symbol_to_file_dict:
                    for symbol in symbol_list:
                        if symbol not in test_symbol_to_file_dict:
                            test_symbol_to_file_dict[symbol] = [random_file]
                        else:
                            test_symbol_to_file_dict[symbol].append(random_file)
                        symbol_to_file_dict[symbol].remove(random_file)

                    del(file_to_symbol_dict[random_file])



                    if random_file not in test_file_to_symbol_dict:
                        test_file_to_symbol_dict[random_file] = symbol_list
                    else:
                        test_file_to_symbol_dict[random_file] = test_file_to_symbol_dict[random_file] + symbol_list

                    break

        else:
            train_dict_priors,test_dict_priors = calc_priors(symbol_to_file_dict, test_symbol_to_file_dict)
            before_KL = KL(train_dict_priors,test_dict_priors)
            after_symbol_to_file_dict = copy.copy(symbol_to_file_dict)
            after_file_to_symbol_dict = copy.copy(file_to_symbol_dict)
            after_test_symbol_to_file_dict = copy.copy(test_symbol_to_file_dict)
            after_test_file_to_symbol_dict = copy.copy(test_file_to_symbol_dict)

            before = copy.copy(after_file_to_symbol_dict[random_file])
            del(after_file_to_symbol_dict[random_file])
            after_test_file_to_symbol_dict[random_file] = before

            for symbol in after_symbol_to_file_dict:
                if random_file in after_symbol_to_file_dict[symbol]:
                    before = copy.copy(after_symbol_to_file_dict[symbol])
                    after_symbol_to_file_dict[symbol].remove(random_file)
                    after_test_symbol_to_file_dict[symbol].append(random_file)



                '''
                                for lab in file_to_symbol_dict[random_file]:


                                    if lab not in after_symbol_to_file_dict:
                                        after_symbol_to_file_dict[lab] = [random_file]
                                    else:
                                        after_symbol_to_file_dict[lab].append(random_file)

                                    after_test_dict_symbols[lab] += 1
                                    after_train_dict_symbols[lab] -= 1
                                    after_sum_test += 1
                                    after_sum_train -= 1
                                '''
            after_train_dict_priors,after_test_dict_priors = calc_priors(after_symbol_to_file_dict, after_test_symbol_to_file_dict)
            after_KL = KL(after_train_dict_priors,after_test_dict_priors)

            if after_KL < before_KL:
                symbol_to_file_dict = after_symbol_to_file_dict
                file_to_symbol_dict = after_file_to_symbol_dict
                test_symbol_to_file_dict = after_test_symbol_to_file_dict
                test_file_to_symbol_dict = after_test_file_to_symbol_dict
                train_dict_priors = after_train_dict_priors
                test_dict_priors = after_test_dict_priors

    return symbol_to_file_dict, file_to_symbol_dict, test_symbol_to_file_dict, test_file_to_symbol_dict, train_dict_priors, test_dict_priors


def KL(train_priors, test_priors):
    sum = 0

    for lab in train_priors:
        if test_priors[lab] != 0:
            try:
                sum += train_priors[lab] * math.log(train_priors[lab] / test_priors[lab] , math.e)
            except ValueError:
                print(test_priors[lab])
        else:
            print('0')
    return sum


def calc_priors(symbol_to_file_dict,test_symbol_to_file_dict):
    sum_train = 0
    sum_test = 0
    train_dict_priors = {}
    test_dict_priors = {}
    for lab in symbol_to_file_dict:
        sum_train += len(symbol_to_file_dict[lab])
    for lab in test_symbol_to_file_dict:
        sum_test += len(test_symbol_to_file_dict[lab])
    for lab in symbol_to_file_dict:
        train_dict_priors[lab] = len(symbol_to_file_dict[lab]) / sum_train

    for lab in test_symbol_to_file_dict:
        test_dict_priors[lab] = len(test_symbol_to_file_dict[lab]) / sum_test

    return train_dict_priors, test_dict_priors

#splits the data into a 70/30 training/testing split while maintaining balanced within each class
#train_dict is a dictionary with symbols as the keys and the UI's associated with them in a list as the values
#returns a dictionary for training and a dictionary for testing
def split_data(train_dict):   
    test_dict = {}
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
    symbol_to_file_dict = {}
    file_to_symbol_dict = {}
    master_dict = {}
    for filename in glob.glob('expressmatch/*.inkml'):
        symbol_to_file_dict, file_to_symbol_dict, master_dict = parse_soup(filename,symbol_to_file_dict, file_to_symbol_dict, master_dict)

    return symbol_to_file_dict,file_to_symbol_dict, master_dict
    
    
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
        #for filename in dct[UI]:
        csv_writer.writerow([UI])

def choose(n, k):
    return math.factorial(n) / (math.factorial(k) * (math.factorial(n-k)))

def bell(n):

    sum = 0

    for k in range(n-1):
        sum += choose(n-1,k) * bell(k)

    return sum
def parse_soup(file, symbol_to_file_dict, file_to_symbol_dict, master_dict):
    with open(file) as f:
        soup = BeautifulSoup(f)
    UI = []
    trace_id_to_pts_list = {}
    trace_id_to_trace_grouping = {}
    file_to_trace_dict = {}
    group_dict ={}
    symbol_to_id = {}
    symbol_to_symbol_id = {}
    trace_id_to_trace = {}
    sym_id_to_trace_id = {}
    sym_id_to_sym = {}
    file_to_trace_id = {}
    sym = 0
    sym_id = 0
    for b in soup.findAll('tracegroup'):

        for child in b.children:
            traceDataRef = []
            for children in child:

                try:
                    if children.name == 'traceview':
                        try:
                            bruh = children["tracedataref"]
                            traceDataRef.append(bruh)
                        except:
                            pass
                        #c = children
                        ele = children.previous_element.previous_element
                        if "String" in str(type(ele)):
                            sym = ele
                            UI.append(ele)
                            if ele in symbol_to_file_dict:
                                symbol_to_file_dict[ele].append(file)

                            else:
                                symbol_to_file_dict[ele] = [file]
                            if sym in symbol_to_id:
                                symbol_to_id[sym].append()
                    else:
                        if children.name == 'annotationxml':
                            sym_id = children['href']

                except:
                    pass


            if len(traceDataRef) != 0:

                if sym in symbol_to_symbol_id:
                    symbol_to_symbol_id[sym].append(sym_id)
                else:
                    symbol_to_symbol_id[sym] = [sym_id]

                sym_id_to_sym[sym_id] = sym

                if file in group_dict:
                    group_dict[file].append([sym, traceDataRef])
                else:
                    group_dict[file] = [[sym, traceDataRef]]

                sym_id_to_trace_id[sym_id] = traceDataRef

            '''
            try:
                string_label = GT_dict[UI].strip('\n')
            except:
                string_label = "invalid"
            if '\\' in string_label:
                string_label = string_label[1:]
                '''

    if file in file_to_symbol_dict:
        file_to_symbol_dict[file] = file_to_symbol_dict[file] + UI
    else:
        file_to_symbol_dict[file] = UI

    tracelist = []
    
    
    for t in soup.findAll('trace'):
        id = t['id']
        cur_points = t.contents[0]
        tokenized = cur_points.split(',')
        
        pts = []
        for token in tokenized:
            xy = token.split()
            pts.append([float(xy[0]),float(xy[1])])

        np_traces =     convert_to_np(pts)
        trace_id_to_trace[id] = np_traces
        if True in np.isnan(np_traces)[:]:
            print('hl')
 
    master_dict[file] = [sym_id_to_sym, symbol_to_symbol_id, sym_id_to_trace_id,trace_id_to_trace]
    return symbol_to_file_dict, file_to_symbol_dict, master_dict

def convert_to_np(pt_pair_list):
    np_ar = np.empty((len(pt_pair_list),2))
    for index, pair in enumerate(pt_pair_list):
        np_ar[index,0] = pair[0]
        np_ar[index,1] = pair[1]
    return np_ar



def copy_train_test(test_file_to_symbol_dict=None, file_to_symbol_dict=None):
    '''
    copies files from test set csv into test_GT folder
    and train_GT folder
    '''

    root_directory = os.path.join(os.environ["HOMEDRIVE"], os.environ["HOMEPATH"], "Documents\\PatternRec\\classifier_project\\HandWriting_Classifier\\")

    #copy files to test_GT folder
    for cur_file in test_file_to_symbol_dict:
        
        filename = cur_file.split('\\')[1].split('.')[0] + '.lg'
        
        src = os.path.join(root_directory, 'all_lg_GT\\', filename)
        dest = os.path.join(root_directory,'test_GT\\', filename)
        
        shutil.copyfile(src, dest)
    
    #copy files to train_GT folder    
    for cur_file in file_to_symbol_dict:
        
        filename = cur_file.split('\\')[1].split('.')[0] + '.lg'
        
        src = os.path.join(root_directory, 'all_lg_GT\\', filename)
        dest = os.path.join(root_directory,'train_GT\\', filename)
    
        shutil.copyfile(src, dest)

    
    
    


def makedirs():
    os.mkdir('train_GT')
    os.mkdir('test_GT')


def main():

    sys.setrecursionlimit(5000)
    makedirs()
    print('hi')
    symbol_to_file_dict, file_to_symbol_dict, master_dict = read_inkml()
    symbol_to_file_dict, file_to_symbol_dict, test_symbol_to_file_dict, test_file_to_symbol_dict,train_dict_priors,test_dict_priors = split_data_KL(symbol_to_file_dict, file_to_symbol_dict)
    


    max = 0

    for key in master_dict:
        for sym in master_dict[key][2]:
            if len(master_dict[key][2][sym]) > max:
                max = len(master_dict[key][2][sym])

    print(max)
    copy_train_test(test_file_to_symbol_dict, file_to_symbol_dict)

    write_to_csv(file_to_symbol_dict, "train.csv")
    write_to_csv(test_file_to_symbol_dict, "test.csv")
    plt.bar(train_dict_priors.keys(), train_dict_priors.values())
    plt.bar(test_dict_priors.keys(), test_dict_priors.values())
    pickle.dump(master_dict, open("master.pkl", "wb"))
    plt.show()



main()