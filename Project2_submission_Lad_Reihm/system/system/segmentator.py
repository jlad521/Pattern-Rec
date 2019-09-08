'''
Justin Lad & Richard (Corey) Riehm
Handwriting Classifier
4/2/2019
'''
import sys
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
import itertools
from queue import *
import copy
from shutil import copyfile
import time 
import joblib
#import math
#from itertools import chain, combinations


#import warnings
#warnings.simplefilter('error')

#open classifier globally, so only have to open once
with open('proj_2-rf.joblib', 'rb') as handle:#change name to newly trained classifier
    #rf_clf = pickle.load(handle)  
    rf_clf = joblib.load(handle)      
    
class_labels = rf_clf.classes_


def segment_classifier(filename, output_path,baseline = False):

    f = open(filename, 'r')

    text_list = f.readlines() 

    start_execution = time.time()
    for file in text_list:
     
        trace_list = parse_soup(file.strip('\n')) #get trace_list from .inkml file
  
        if baseline: 

            segmentation_prediction = baseline_segmenter(trace_list)
            write_LG_output(segmentation_prediction,file,output_path)

        else:

            start = time.time()
            trace_lu = {}
            trace_centroids = get_trace_centroids(trace_list)

            trace_lu = get_n_grouping(trace_list, trace_centroids,1,trace_lu) # single pts

            if len(trace_list) > 1:
                trace_lu = get_n_grouping(trace_list, trace_centroids,2,trace_lu) # d pts

            if len(trace_list) >2:
                trace_lu = get_n_grouping(trace_list, trace_centroids,3,trace_lu) # trip pts

            if len(trace_list) > 3:
                trace_lu = get_n_grouping(trace_list, trace_centroids,4,trace_lu) # quad pts

            #valid_groupings = [key for key in trace_lu]
            path = greedy(trace_lu,len(trace_list))
            write_LG_output(path,file,output_path)

            end = time.time() - start
            #plt_points(trace_list)
            #print(f'took {end} amount of time to get stroke groupings for {file}')


    end = time.time() - start_execution
    print(f'took {end} amount of time to test all files')

        

def mark_used(stroke_tuple,used):
    for stroke_grp in stroke_tuple:
        #for stroke_id in stroke_grp:
        used[stroke_grp] = True
    return used

def check_full(used):
    for stroke in used:
        if  not stroke: #if false (stroke not used)
            return False
    return True

def valid_move(stroke_tuple,used):
    for stroke_grp in stroke_tuple:
        #for stroke_id in stroke_grp:
        if used[stroke_grp]: return False
    return True

def greedy(trace_lu,num_trace):

    used = np.zeros(num_trace,dtype=bool)
    prob_array = np.empty(len(trace_lu))
    prob_lookup = {}

    #represent trace_lookups as probabilities
    for i, key in enumerate(trace_lu):
        prob_array[i] = trace_lu[key][0]
        prob_lookup[i] = key


    cur_path = []
   
    while not check_full(used):
        selected = False
        while not selected:
            ind = np.argmax(prob_array)
            correspond_stroke_group = prob_lookup[ind]
            if valid_move(correspond_stroke_group,used):
                mark_used(correspond_stroke_group,used)
                cur_path.append(correspond_stroke_group)                
                selected = True

            prob_array[ind] = 0 #dont look at this again

    final_prediction = {} 
    for group in cur_path:
        final_prediction[group] = trace_lu[group][1]

    return final_prediction

def baseline_segmenter(traces):
    #segments everything as single trace
    single_trace_lu = {}

    #classifying one at a time right now, can do bulk later
    for i, trace in enumerate(traces):
        preprocess([trace])

        features = np.array([get_features([trace])])
        data = np.nan_to_num(features)
        predict_probs = rf_clf.predict_proba(data)
        best_guess = np.argmax(predict_probs)

        single_trace_lu[tuple([i])] = class_labels[best_guess]         
        
    return single_trace_lu


def write_LG_output(all_files,filen,folder_ext):
    #assume all_files is single dict for now
    cur_file = all_files #this should take a list of dictionaries in the future
    
    
    filename = filen.split('\\') #cut off the inkml part
    output_filename = filename[1].split('.')[0] + '.lg'
    
    #set desired path output
    filenam = os.path.join(os.environ["HOMEDRIVE"], os.environ["HOMEPATH"], "Documents\\PatternRec\\classifier_project\\HandWriting_Classifier\\",folder_ext, output_filename)
    csv_writer = csv.writer(open(filenam,'w',newline=''))

    used_symbols = {}
    for trace_group in cur_file:
       
        sym = cur_file[trace_group]
        if sym not in used_symbols:
            sym_id = sym + '_1'
            used_symbols[sym] = 2
        else:
            sym_id = sym + '_' + str(used_symbols[sym]) 
            used_symbols[sym] += 1
        to_write = ['O',sym_id, sym, '1.0']
        
        #cur_traces = list(trace_group)
        for trace in trace_group:        
            to_write.append(trace)
        csv_writer.writerow(to_write)
        




def get_n_grouping(trace_list, centroids,n,trace_lu):
    copy_trace_list = trace_list.copy()
    #trace_lu = {}
    unique_groupings = set()

    for stroke_id, trace in enumerate(trace_list):
        if n > 1:
            grouping = get_closest_k_neighbors(stroke_id,centroids,n) #returns itself in list
            grouping.sort() #list of stroke ids in group
            sorted_group = tuple(grouping)
            
        else: sorted_group = (stroke_id,)

        if not sorted_group in unique_groupings: #only compute once

            unique_groupings.add(tuple(sorted_group)) 

            grp_trace_list = [copy_trace_list[sorted_group[i]].copy() for i in range(n)]

            preprocess(grp_trace_list)

            features = [get_features(grp_trace_list)]
            no_error_features = np.nan_to_num(features)

            predict_probs = rf_clf.predict_proba(no_error_features)
            best_guess = np.argmax(predict_probs)

            trace_lu[sorted_group] = [predict_probs[0][best_guess],class_labels[best_guess]]
            #trace_lu[tuple(sorted_group)] = [predict_probs[best_guess], class_labels[best_guess]]

    return trace_lu

'''
FOR USE IF BRUTE FORCE APPROACH
def min_cost_perm(perm_list,prob_lookup):
    min_cost = 999999999
    best_perm = None
    for perm in perm_list:
        cur_loss = 0
        for grp in perm:
            cur_loss += prob_lookup[grp] * len(grp) #increase cost by the length, so it is comparable to single stroke loss

        if cur_loss < min_cost:
            min_cost = cur_loss
            best_perm = perm

    return best_perm
'''





def rec_combos(d, valid, l, c):

    r = []
    if len(d) < 1:
        return []
    if len(d) < 2:
        return d[0]
    

    
    if l>= 4 and len(d) >=4:
        for i in range(len(d)):

            for j in range(i + 1, len(d)):
                for k in range(j+1, len(d)):
                    for m in range(k+1, len(d)):
                        
                        if len(d) == 4:
                            r.append(d[0])
                            r.append(d[1])
                            r.append(d[2])
                            r.append(d[3])
                            #r.append(rec_combos(d, valid, l-1))
                        remaining = d.copy()        
                        del remaining[m]
                        del remaining[k]
                        del remaining[j]
                        del remaining[i]
                        if len(remaining) > 0:
                            #if (d[i], d[j], d[k], d[m]) in valid:
                            for n in range(l):
                                r.append([[d[i], d[j], d[k], d[m]], rec_combos(remaining, valid, l-n,c)])   
                                
                                #c[0]+=1
                                #r.append([[d[i], d[j], d[k], d[l]], rec_combos(remaining, valid, l,c)])
                                #c[0]+=1
                                #r.append([d[i], d[j], d[k], d[l], rec_combos(remaining, valid, l-1,c)])
                                

    elif l>= 3 and len(d) >= 3:
        for i in range(len(d)):

            for j in range(i + 1, len(d)):
                for k in range(j+1, len(d)):
                
                    if len(d) == 3:
                        r.append([d[0], d[1], d[2]])
                        #r.append(d[1])
                        #r.append(d[2])
                        #r.append(rec_combos(d, valid, l-1))
                    remaining = d.copy()        
                    del remaining[k]
                    del remaining[j]
                    del remaining[i]
                    if len(remaining) > 0:
                        #if (d[i], d[j], d[k]) in valid:
                        for n in range(l):
                            r.append([[d[i], d[j], d[k]], rec_combos(remaining, valid, l-n,c)])   

    if l >= 2 and len(d) >= 2:
        for i in range(len(d)):

            for j in range(i+1, len(d)):
                
                if len(d) == 2:
                    r.append([d[0],d[1]])
                    #r.append(d[1])
                    #r.append(rec_combos(d, valid, l-1))
                remaining = d.copy()        
                del remaining[j]
                del remaining[i]
                if len(remaining) > 0:
                    #if (d[i], d[j]) in valid:
                    for n in range(l):
                        r.append([[d[i], d[j]], rec_combos(remaining, valid, l-n,c)]) 
    else:
        return d
        #for i in range(1, len(d)):
            #remaining = d.copy()
            #del remaining[i]
            #r.append([d[i], rec_combos(remaining, valid, l,c)])
    
    return r
    
            

    

def get_closest_k_neighbors(neighbor_id, centroids,k):
    centroid_list = []
    
    for key in centroids:
        centroid_list.append(centroids[key]) 
        
    cur_x = centroids[neighbor_id][0]
    cur_y = centroids[neighbor_id][1]

    
    q = PriorityQueue()
    for i in range(len(centroid_list)):
        #if i == neighbor_id: continue
        cur_pt = [centroid_list[i][0],centroid_list[i][1]]
        cur_dist = distance([cur_x,cur_y],cur_pt)
        q.put((cur_dist,i))

    closest_traces = []
    
    for i in range(k):
        v = q.get()
        closest_traces.append(v[1])
        
    return closest_traces



#gets the top 10 predictions from the RF output matrix
def get_rf_guesses(predictions_matrix,class_labels):
    k = 10
    guess_list = []
    for row in predictions_matrix:
        top_guesses = np.argpartition(row, -k)[-k:]
        top_guesses = top_guesses[np.argsort(row[top_guesses])]        
        cur_guesses = []
        for i in range(k-1,-1,-1):
            cur_guesses.append(class_labels[top_guesses[i]])
            
        guess_list.append(cur_guesses)
    return guess_list
   
       

#RF-Classifier
# takes a test set and returns a list of predictions based off the
#pickled trained KD-Tree
def rf_classifier(test, tree_specifier):
    #truth_pkl= open(tree_specifier + '-rf.pkl', 'rb')
    truth_pkl= open('truth-rf.pkl', 'rb')
    
    truth = pickle.load(truth_pkl)  
   
    test = np.nan_to_num(test)
    
    probability_predictions = truth.predict_proba(test)
    
    class_labels = truth.classes_
    predictions = get_rf_guesses(probability_predictions,class_labels)    
    return predictions

     
#takes a symbol and applies preprocessing functions to it    
def preprocess(trace_list):
    normalize(trace_list)
    smooth(trace_list)
    smooth(trace_list)
    #deduplicate(trace_list)
    #centroids = get_trace_centroids(trace_list)

    
#returns euclidian distances between two points
def distance(pt_a,pt_b):
    #assumes 2D:
    return math.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)

def compute_centroid(trace):
    trace = np.array(trace)

    x_max, x_min = np.amax(trace[:,0]), np.amin(trace[:,0])
    y_max, y_min = np.amax(trace[:,1]), np.amin(trace[:,1])  
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    return [x_center, y_center]
    

    #change name to get centroids later
def get_trace_centroids(traces):
    #compute centroid of each trace
    centroid_lookup = {}
    for index, trace in enumerate(traces):
        centroid_lookup[index] = compute_centroid(trace)
        
    return centroid_lookup


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

    all_traces = []
    angles = []

    for trace in trace_list:
        for t in trace:
            all_traces.append(t)
        #all_traces = all_traces + trace
    
    #for trace in trace_list:"
    trace = all_traces
  
    #angle_bins = np.zeros(len(trace_list))
    
    #for trace in trace_list:
    origin = compute_centroid(trace)
    reference_pt = [1.0,origin[1]] #line due east
    for i in range(len(trace)):
        opp = distance(origin,reference_pt)
        adj = distance(origin,trace[i])
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


#removes points that are extremely close to each other    
def deduplicate(symbol):
    #print('start')
    #plt_points(symbol)

    #for index, s in enumerate(symbol):
    for index in range(len(symbol)):
        start_len = len(symbol[index])

        #for i in range(len(s) - 1):
        i = 1
        while i < len(symbol[index])-1:
            if(distance(symbol[index][i], symbol[index][i+1]) < .05):
    
                symbol[index] = np.delete(symbol[index], symbol[index][i],axis=0)
                #i -= 1
            i += 1

        end_len = len(symbol[index])
        if start_len != end_len: print(start_len - end_len)

    #print('done')
    plt_points(symbol)
    
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
    
    if y_max == y_min: 
        #print_flag = True
        y_max += .1
    if x_max == x_min: 
        #print_flag = True
        x_max += .1

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

#parses the inkml file using Beautiful Soup, returns the list of traces/points with the label
def parse_soup(file):
    trace_list = []
    with open(file) as f:
        soup = BeautifulSoup(f,features='html.parser')
    '''
    for b in soup.findAll('annotation'):
        if b['type'] == 'UI':
            label = b.contents[0].strip('\n')
    '''
    for t in soup.findAll('trace'):
        cur_points = t.contents[0]
        tokenized = cur_points.split(',')
        pts = []
        for token in tokenized:
            xy = token.split()
            pts.append([float(xy[0]),float(xy[1])])
        trace_list.append(convert_to_np(pts))
        
    return trace_list


def create_label_lookup(truth_dict):
    label_lookup = {}
    int_label = 0
    for key in truth_dict:
        label_lookup[key] = int_label
        int_label += 1
    return label_lookup 

#gets the x coordinate at the first point in the first trace
def min_x(symbol):
    x_min = sys.maxsize
    for s in symbol:
        x_min = min(np.amin(s[:,0]),x_min)
    return x_min
    #return symbol[0][0][0]

#gets the y coordinate at the first point in the first trace
def min_y(symbol):
    y_min = sys.maxsize
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
        
    if x_max - x_min < 0.00000001:  #TODO: tune up errors
        #print('hit thresh')
        return 1
    
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
    if len(c[0]) < 2:
        return 0,0,0
    #try:
    cov = np.cov(c)
    #except:
    #    return 0,0,0

    return cov[0][0], cov[0][1], cov[1][1]



#calls all of the feature extraction methods on a symbol
#returns the list of features
def get_features(symbol_np_list):
    get_angles(symbol_np_list)
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

    
def plt_points(lst_of_arrays,Flip = False):
    for i in range(len(lst_of_arrays)):
        demo_plt = lst_of_arrays[i]
        if Flip:
            flip_y = -1*(demo_plt[:,1]-1)
            plt.scatter(demo_plt[:,0],flip_y)
        else:
            plt.scatter(demo_plt[:,0],demo_plt[:,1])
    plt.show()
    

def makedirs():
    os.mkdir('predictions_train_baseline')
    os.mkdir(' predictions_train_greedy')
    os.mkdir('predictions_test_baseline')
    os.mkdir('predictions_test_greedy')

def main():
    makedirs()

    segment_classifier('test.csv', 'predictions_test_baseline\\',True) #baseline segmenter
    segment_classifier('test.csv', 'predictions_test_greedy\\') #greedy segmenter

    segment_classifier('train.csv', 'predictions_train_baseline\\',True) #baseline segmenter
    segment_classifier('train.csv', 'predictions_train_greedy\\') #greedy segmenter

    valid = [(1,3),(3,4), (2,5)]
    d = [1, 2, 3,4,5,6]
    c = [0]
    r = rec_combos(d, valid, 4,c)

    for i in range(len(r)):
        #print(i)
        print(r[i])

    #print(r)

    print(c[0])



    #comb()
    #combinations()
    
    
    
    
    

main()

