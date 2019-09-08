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
from joblib import dump,load
  
#reads a csv file containing filenames and parses each inkml file, retrieving the label associated and the collection of traces and points
#the points are then subjected to the preprocessing methods and the features are then extracted
# returns the matrix of features for each symbol in the file, along with a label associated with each file
def read_csv(file, GT_dict):
    csv_reader = open(file,'r')
    text_list = csv_reader.readlines()
    feature_matrix = []
    labels_matrix = []    
    for filen in text_list:
        pts_list,label = parse_soup(filen, GT_dict)
        preprocess(pts_list)
        feature_matrix.append(get_features(pts_list)) #ignore labels 
        labels_matrix.append(label)    
       
    
    #create an n x f array for input to kd tree, n = num files, f = num features
    return [np.array(feature_matrix), labels_matrix]

#takes a symbol and applies preprocessing functions to it    
def preprocess(pts_list):
    normalize(pts_list)
    smooth(pts_list)
    smooth(pts_list)
    deduplicate(pts_list)

def read_inkml(filename):
    #read_truth_file() # use global var
    truth_pkl = open('truth_GT.pkl', 'rb')
    truth_GT_dict = pickle.load(truth_pkl)
    both_pkl = open('both_GT.pkl', 'rb')
    both_GT_dict = pickle.load(both_pkl)
    
    #unpickle_objects()
    
    both_train_features = read_csv(filename,both_GT_dict)

    return both_train_features


def convert_labels(labels):
    kd_index_lookup = {}
    for index,symbol_class in enumerate(labels):
        kd_index_lookup[index] = symbol_class
    return kd_index_lookup

#gets the mean x value for the symbol
def mean_x(symbol):
    sum = 0
    count = 0
    for t in symbol:
        for coord in t:
            sum += coord[0]
            count += 1

    return sum/count

#gets the mean y value for the symbol
def mean_y(symbol):
    sum = 0
    count = 0
    for t in symbol:
        for coord in t:
            sum += coord[1]
            count += 1

    return sum / count

#gets the covariance matrix for the symbol, returns xx, xy, and yy covariances
def covariance(symbol):
    pts = []
    for t in symbol:
        for coord in t:
            pts.append(coord)

    c = np.array(pts).T
    cov = np.cov(c)

    return cov[0][0], cov[0][1], cov[1][1]

#gets the x coordinate at the first point in the first trace
def start_x(symbol):
    return symbol[0][0][0]

#gets the y coordinate at the first point in the first trace
def start_y(symbol):
    return symbol[0][0][1]
#gets the x coordinate at the last point in the last trace
def end_x(symbol):
    return symbol[-1][-1][0]

#gets the y coordinate at the last point in the last trace
def end_y(symbol):
    return symbol[-1][-1][1]

#gets the height-width aspect ratio for the symbol
def get_h_w_ratio(symbol):
    x_max = -sys.maxsize
    x_min = sys.maxsize
    y_max = -sys.maxsize
    y_min = sys.maxsize
    for s in symbol:
        x_max = max(np.amax(s[:,0]),x_max)
        x_min = min(np.amin(s[:,0]),x_min)
        y_max = max(np.amax(s[:,1]),y_max)
        y_min = min(np.amin(s[:,1]),y_min)
    
    return (y_max-y_min) / (x_max - x_min)
                  
#gets the total distance between adjacent points
def get_distance(symbol):
    #need to test
    cur_dist = 0
    for s in symbol:
        for row in range(1,len(s)):
            cur_pt = np.array([s[row,0],s[row,1]])
            past_pt = np.array([s[row-1,0],s[row-1,1]])
            cur_dist += distance(cur_pt, past_pt)
    return cur_dist

#returns euclidian distances between two points
def distance(pt_a,pt_b):
    #assumes 2D:
    return math.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)

#returns the number of times the symbol intersects a centered vertical line
def intersect_y(symbol):
    sum = 0
    for t in symbol:
        for i in range (len(t) - 1):
            if(intersect([.5, 0], [.5, 1], t[i], t[i+1])):
                sum+= 1

            #sum += line_intersection(((.5, 1),(.5, 0)), (t[i], t[i+1]))

    return sum

#returns the number of times the symbol intersects a centered horizontal line
def intersect_x(symbol):
    sum = 0
    for t in symbol:
        for i in range(len(t) - 1):
            if (intersect([0, .5], [1, .5], t[i], t[i + 1])):
                sum += 1

            # sum += line_intersection(((.5, 1),(.5, 0)), (t[i], t[i+1]))

    return sum

#functions for checking if two line segments intersect
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#returns number of points in a symbol
def get_num_pts(symbol):
    count = 0
    for s in symbol:
        count += 1
    return count

#calls all of the feature extraction methods on a symbol
#returns the list of features
def get_features(symbol_np_list):
    num_features = 15
    cur_row = np.empty(num_features)
    cur_row[0] = len(symbol_np_list)
    cur_row[1] = get_h_w_ratio(symbol_np_list)
    cur_row[2] = get_distance(symbol_np_list)
    cur_row[3] = start_x(symbol_np_list)
    cur_row[4] = start_y(symbol_np_list)
    cur_row[5] = end_x(symbol_np_list)
    cur_row[6] = end_y(symbol_np_list)
    cur_row[7] = intersect_y(symbol_np_list)
    cur_row[8] = intersect_x(symbol_np_list)
    cur_row[9] = mean_x(symbol_np_list)
    cur_row[10] = mean_y(symbol_np_list)
    cur_row[11], cur_row[12], cur_row[13] = covariance(symbol_np_list)
    cur_row[14] = distance([cur_row[3],cur_row[4]],[cur_row[5],cur_row[6]])
    return cur_row


#random forest classifier
def random_forrest(data,labels, filename):
    rf = RandomForestClassifier(n_estimators=25, max_depth=25, criterion='entropy', random_state=4224)
   
    data = np.nan_to_num(data)
    rf.fit(data,labels)
    
    pkl = open(filename + '.pkl', 'wb')
    pickle.dump(rf,pkl)
    
#kd-tree classifier
def kd_tree(data,labels, filename):
    data = np.nan_to_num(data)
    tree = KDTree(data)
  
    pkl = open(filename + ".pkl", 'wb')
    pickle.dump(tree,pkl)
    #kd index lookup holds the index of each class and associated symbol. This way, can lookup the 
    kd_index_lookup = convert_labels(labels)
    pkl_data = open(filename + '-data.pkl','wb')
    pickle.dump(kd_index_lookup,pkl_data)
  
        

def plt_points(lst_of_arrays,Flip = False):
    for i in range(len(lst_of_arrays)):
        demo_plt = lst_of_arrays[i]
        if Flip:
            flip_y = -1*(demo_plt[:,1]-1)
            plt.scatter(demo_plt[:,0],flip_y)
        else:
            plt.scatter(demo_plt[:,0],demo_plt[:,1])
    plt.show()
    #print('
    print('------------That was all traces in that inkml file-------------')



'''
for future implementation
def resample(symbol):
    num_pts = 50
    symbol_length = 0
    
  
    for t in symbol:
        for index,coord in enumerate(t):
            if index == len(t)-1: continue
            symbol_length += distance(coord,t[index+1])
            
    pt_dist_threshold = symbol_length / num_pts

    alpha = .05
    
    for index, t in enumerate(symbol):
        N = len(t) - 1
        new_pts = []
        L = [None for _ in range(len(t))]
        L[0] = 0
        for i in range(1,len(t)):
            L[i] = L[i-1] + distance(t[i], t[i-1])
        
        m = math.floor(L[N-1] / alpha)
        
        new_pts.append(t[0])
        j = 1
        for p in range(1,m - 2):
            while L[j] < p * alpha:
                j += 1
            C = (p * alpha - L[N-1]) / (L[N] - L[N-1])
            new_pts.append([t[N-1, 0] + (t[N, 0] - t[N-1, 0]) * C, t[N-1, 1] + (t[N, 1] - t[N-1, 1]) * C])
            
        new_pts.append([t[N,0], t[N,1]])
        
        symbol[index] = np.array(new_pts)
    
    for p in range(num_pts-2):
        for t in symbol:
            dist_trav = 0
            for index, coord in enumerate(t):
                if index == len(t)-1: continue
            
                dist_trav += distance(coord,t[index+1])
                if dist_trav >= pt_dist_threshold:
                    dist_trav = 0
                    new_pts.append([(coord[0] + t[index+1][0])/2, (coord[1] + t[index+1][1])/2])
                    t = np.array(new_pts)                            
    

'''

#removes points that are extremely close to each other    
def deduplicate(symbol):

    
    for index, s in enumerate(symbol):
        
        for i in range(len(s) - 1):
            if(distance(s[i], s[i+1]) < .05):
    
                np.delete(s, s[i+1])
                i -= 1
    
#smooth the points in a symbol
def smooth(symbol):
    for s in symbol:
        for index,coord in enumerate(s):
            if index == 0 or index == len(s)-1: continue
            s[index,0] = (s[index-1,0] + s[index,0] + s[index+1,0])/3
            s[index,1] = (s[index-1,1] + s[index,1] + s[index+1,1])/3

#normalizes the symbol to [0,1] range
def normalize(symbol):
    x_max,y_max = -1 * sys.maxsize,  -1 * sys.maxsize
    x_min,y_min = sys.maxsize, sys.maxsize
    
    for s in symbol:
        x_max = max(np.amax(s[:,0]),x_max)
        x_min = min(np.amin(s[:,0]),x_min)
        y_max = max(np.amax(s[:,1]),y_max)
        y_min = min(np.amin(s[:,1]),y_min)
    
    for s in symbol:
        for index, coord in enumerate(s):
            coord[0] = (coord[0] - x_min) / (x_max - x_min)
            coord[1] = -1*(((coord[1] - y_min) / (y_max - y_min))-1)
            

def convert_to_np(pt_pair_list):
    np_ar = np.empty((len(pt_pair_list),2))
    for index, pair in enumerate(pt_pair_list):
        np_ar[index,0] = pair[0]
        np_ar[index,1] = pair[1]
    return np_ar

#parses a inkml file and returns list of points and the UI
def parse_soup(file,GT_dict):
    pts_list = []
    with open(file.strip('\n')) as f:
        soup = BeautifulSoup(f)
    
    for b in soup.findAll('annotation'):
        if b['type'] == 'UI':
            label = b.contents[0].strip('\n')
            try:
                string_label = GT_dict[label].strip('\n')
            except:
                string_label = "invalid"
                
            if '\\' in string_label:
                string_label = string_label[1:]
            
    for t in soup.findAll('trace'):
        cur_points = t.contents[0]
        tokenized = cur_points.split(',')
        pts = []
        for token in tokenized:
            xy = token.split()
            pts.append([float(xy[0]),float(xy[1])])
        pts_list.append(convert_to_np(pts))
        
    return pts_list,string_label
    


def main():
    
   
    both_train_features = read_inkml('both_train.csv')
    
    #kd_data,rf_data,labels = read_inkml()
    #kd_tree(truth_train_features[0],truth_train_features[1], "truth-kd-tree")
    kd_tree(both_train_features[0],both_train_features[1], "both-kd-tree")
    #random_forrest(truth_train_features[0],truth_train_features[1], "truth-rf")
    random_forrest(both_train_features[0],both_train_features[1], "both-rf")
    '''
    both_train_features = read_inkml('bonus_train.csv')
    kd_tree(both_train_features[0],both_train_features[1], "bonus-kd-tree")
    random_forrest(both_train_features[0],both_train_features[1], "bonus-rf")
     '''
main()
