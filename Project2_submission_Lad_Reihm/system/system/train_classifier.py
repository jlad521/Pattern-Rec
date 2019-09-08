'''
Justin Lad & Richard (Corey) Riehm
Handwriting Classifier
4/2/2019
'''
#from Kruskal import Graph
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
#from joblib import dump,load
import time
import joblib
import math
import copy

with open('master.pkl', 'rb') as handle:
    master_dict = pickle.load(handle)


  
#reads a csv file containing filenames and parses each inkml file, retrieving the label associated and the collection of traces and points
#the points are then subjected to the preprocessing methods and the features are then extracted
# returns the matrix of features for each symbol in the file, along with a label associated with each file
def read_csv(file):
    csv_reader = open(file,'r') 
    text_list = csv_reader.readlines()
    feature_matrix = []
    labels_matrix = []    
    
    for filen in text_list:
        #pts_list, label = parse_soup(filen, GT_dict)
        traces_to_symbol_dict = parse_soup(filen) 

        
        #preprocess(cur_pts_list)    
        
        for key in traces_to_symbol_dict:
            cur_pts_list = []
            #all_pts = []
            for traceID in key:
                #print(master_dict[filen.strip('\n')][3])
                cur_pts_list.append(master_dict[filen.strip('\n')][3][traceID])
                #all_pts += master_dict[filen.strip('\n')][3][traceID]
                
            preprocess(cur_pts_list)    
            feature_matrix.append(get_features(cur_pts_list)) #append to feature matrix 
            labels_matrix.append(traces_to_symbol_dict[key])
        '''
        for symbol in symbols_in_file:
            
            feature_matrix.append(get_features(pts_list)) #ignore labels 
            labels_matrix.append(label) 
       '''
    
    #create an n x f array for input to kd tree, n = num files, f = num features
    return [np.array(feature_matrix), labels_matrix]

#takes a symbol and applies preprocessing functions to it    
def preprocess(pts_list):
    normalize(pts_list)
    smooth(pts_list)
    smooth(pts_list)
    #deduplicate(pts_list)
    #build_MST(pts_list)
    #plt_points(pts_list)
    
    
    

def read_inkml(filename):

    both_train_features = read_csv(filename)

    return both_train_features

#parses a inkml file and returns list of points and the UI
def parse_soup(file):
    pts_list = []
    #with open(file.strip('\n')) as f:
    #    soup = BeautifulSoup(f)
        
    file_dict = master_dict[file.strip('\n')]
    
    file_info = {}
    #for key in file_dict[1]:
        
        
    for key in file_dict[2]: #sym_ID_to_trace_id 
        cur_traces = tuple(file_dict[2][key])
        file_info[cur_traces] = file_dict[0][key]
        
        #pts_list.append(file_dict[3][key])
    
    #print('hi')
    '''
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
    ''' 
    return file_info
    #return pts_list, test_file_to_symbol_dict[file]



def compute_centroid(trace):
    trace = np.array(trace)
    x_max, x_min = np.amax(trace[:,0]), np.amin(trace[:,0])
    y_max, y_min = np.amax(trace[:,1]), np.amin(trace[:,1])  
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    return [x_center, y_center]
    


def bin_angles(angles):

    bins = [45*i for i in range(8)]
    bin_count = [0 for i in range(8)]
    #count = 0
    #midpt = 45.0/2.0
    for angle in angles:
        min_dst = 99999
        for i, cur_bin in enumerate(bins):
            cur_dist = abs(angle - cur_bin)
            if cur_dist < min_dst:
                assigned_bin = i
                min_dst = cur_dist
        #print(bins[assigned_bin])
        bin_count[assigned_bin] = bin_count[assigned_bin] +1

    return bin_count
        

def get_angles(trace_list=None):

    #angle_bins = np.zeros(len(trace_list))
    angles = []

    all_traces = []


    for trace in trace_list:
        for t in trace:
            all_traces.append(t)
        #all_traces = all_traces + trace
    
    #for trace in trace_list:"
    trace = all_traces
    origin = compute_centroid(trace)
    reference_pt = [1.1,origin[1]] #line due east (0 angle)
    for i in range(len(trace)):
        opp = distance(origin,reference_pt)
        adj = distance(origin,trace[i])   #if origin == trace[i], aka single point trace, this a problem
        hyp = distance(trace[i],reference_pt)

        if opp == 0 or adj == 0  : 
            #print('it happened')
            continue
        
        numer = (opp**2 + adj**2 - hyp**2)
        angle = math.cosh(numer/(2*opp*adj)) * (180.0/math.pi)

        if trace[i][1] < origin[1]: #if angle greater than 180 degree 
            angle = (180 - angle) + 180

        angles.append(angle)

    binned_angles = bin_angles(angles)
    
    #print(binned_angles)
    #print(angles)

    return binned_angles



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
    if len(c[0]) <2:
        return 0,0,0
    cov = np.cov(c)
    #except:
    #    return 0,0,0

    return cov[0][0], cov[0][1], cov[1][1]

#gets the x coordinate at the first point in the first trace
def min_x(symbol):
    x_min = 9999999
    for s in symbol:
        x_min = min(np.amin(s[:,0]),x_min)
    return x_min
    #return symbol[0][0][0]

#gets the y coordinate at the first point in the first trace
def min_y(symbol):
    y_min = 999999
    for s in symbol:
        y_min = min(np.amin(s[:,1]),y_min)
    return y_min
    #return symbol[0][0][1]
#gets the x coordinate at the last point in the last trace
def max_x(symbol):
    x_max = -sys.maxsize
    for s in symbol:
        x_max = max(np.amax(s[:,0]),x_max)
    return x_max
    #return symbol[-1][-1][0]

#gets the y coordinate at the last point in the last trace
def max_y(symbol):
    y_max = -sys.maxsize
    for s in symbol:
        y_max = max(np.amax(s[:,1]),y_max)
    return y_max
    #return symbol[-1][-1][1]

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
    
    if x_max - x_min < 0.00000001: return 1
    
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
    num_features = 23
    cur_row = np.empty(num_features)
    cur_row[0] = len(symbol_np_list)
    cur_row[1] = get_h_w_ratio(symbol_np_list)
    cur_row[2] = get_distance(symbol_np_list)
    cur_row[3] = min_x(symbol_np_list)
    cur_row[4] = min_y(symbol_np_list)
    cur_row[5] = max_x(symbol_np_list)
    cur_row[6] = max_y(symbol_np_list)
    cur_row[7] = intersect_y(symbol_np_list)
    cur_row[8] = intersect_x(symbol_np_list)
    cur_row[9] = mean_x(symbol_np_list)
    cur_row[10] = mean_y(symbol_np_list)
    cur_row[11], cur_row[12], cur_row[13] = covariance(symbol_np_list)
    cur_row[14] = distance([cur_row[3],cur_row[4]],[cur_row[5],cur_row[6]])
    angle_buckets = get_angles(symbol_np_list)
    for i, bucket in enumerate(angle_buckets):
        cur_row[15+i] = bucket
    return cur_row


#random forest classifier
def random_forest(data,labels, filename):
    rf = RandomForestClassifier(n_estimators=100, max_depth=25, criterion='entropy', random_state=4224)
   
    data = np.nan_to_num(data)
    rf.fit(data,labels)
    
    #pkl = open(filename + '.joblib', 'wb')
    joblib.dump(rf,filename+'.joblib',compress=3)
    #pickle.dump(rf,pkl)
    
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



#removes points that are extremely close to each other    
def deduplicate(symbol):

    
    for s in symbol:
        
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
    np.seterr(all='raise')

    #original = copy.deepcopy(symbol)

    x_max,y_max = -1 * sys.maxsize,  -1 * sys.maxsize
    x_min,y_min = sys.maxsize, sys.maxsize
    
    for s in symbol:
        x_max = max(np.amax(s[:,0]),x_max)
        x_min = min(np.amin(s[:,0]),x_min)
        y_max = max(np.amax(s[:,1]),y_max)
        y_min = min(np.amin(s[:,1]),y_min)

    #print_flag = False
    
    if y_max == y_min: 
        #print_flag = True
        y_max += .1
    if x_max == x_min: 
        #print_flag = True
        x_max += .1
    for s in symbol:
        for coord in s:
            coord[0] = (coord[0] - x_min) / (x_max - x_min)
            coord[1] = -1*(((coord[1] - y_min) / (y_max - y_min))-1)
            #if math.isnan(coord[1]) or math.isnan(coord[0]):
            #    continue
               #print('easy there')
    #if print_flag:
    #    plt_points(original)
    #    plt_points(symbol)

def convert_to_np(pt_pair_list):
    np_ar = np.empty((len(pt_pair_list),2))
    for index, pair in enumerate(pt_pair_list):
        np_ar[index,0] = pair[0]
        np_ar[index,1] = pair[1]
    return np_ar


    


def main():

    start = time.time()
    train_set_features = read_inkml('train.csv')
    random_forest(train_set_features[0],train_set_features[1], "proj_2-rf")
    end = time.time() - start
    print(f' took {end} amount of time to train')

    
    #both_train_features = read_inkml('bonus_train.csv')
    #kd_tree(both_train_features[0],both_train_features[1], "bonus-kd-tree")
    
    
    
    
    
main()
