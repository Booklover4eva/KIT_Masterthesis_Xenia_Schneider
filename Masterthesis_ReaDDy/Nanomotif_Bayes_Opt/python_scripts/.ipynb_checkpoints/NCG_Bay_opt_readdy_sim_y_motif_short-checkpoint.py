#Setting up and running Bayesian Optimization of Y-motif simulation in readdy (can be used for other simulations)
#functions used in NCG_param_opt_y_motif_BoTorch notebook
#Updated for short Y-motif

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import readdy
import math
import scipy
import itertools
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import time
import scipy.integrate as integrate
#print(readdy.__version__)
from scipy.spatial.transform import Rotation as Rot
#from sklearn.cluster import DBSCAN
#from sklearn import metrics
from collections import Counter
import random
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d

from multiprocessing import Process, Manager
from functools import partial

import contextlib
import io



##########
#General functions
##########

#run n functions in a list [f_1,...f_n] in parallel with parameters [[params_1],...,[params_n]]
def run_p(fcs,params):
    proc = []
    for i in range(len(fcs)):
        p = Process(target=fcs[i], args=(*params[i],))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

        
#run n functions  in a list [f_1,...f_n] in parallel with parameters [[params_1,return_dict],...,[params_n,return_dict]]
#add ouput in return_dict 
def run_p2(fcs,params):
    proc = []
    manager = Manager()
    return_dict = manager.dict()
    for i in range(len(fcs)):
        p = Process(target=fcs[i], args=(*params[i],return_dict))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()   
    return return_dict

        
#run n functions in a list [f_1,...f_n] in parallel with parameters [{params_1},...,{params_n}]
def run_p_dict(fcs,params):
    proc = []
    for i in range(len(fcs)):
        p = Process(target=fcs[i], kwargs=params[i])
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
        
#run n functions  in a list [f_1,...f_n] in parallel with parameters [{params_1,return_dict},...,{params_n,return_dict}]
#add ouput in return_dict 
def run_p2_dict(fcs,params):
    proc = []
    manager = Manager()
    return_dict = manager.dict()
    for i in range(len(fcs)):
        p = Process(target=fcs[i], kwargs={**params[i], 'return_dict': return_dict})
        p.start()
        proc.append(p)
    for p in proc:
        p.join()   
    return return_dict

#function that takes a lists of lists and generates an array of all possible combinations of entries in the lists
#e.g: [[1,2],[3,4]] -> [[[1, 3],[1, 4]],[[2, 3],[2, 4]]]  (shape=2,2,2)
def gen_all_comb_from_list_of_lists(input_list):
    #generate all possible combinations using itertools.product
    combinations = list(itertools.product(*input_list))

    #convert the combinations to a NumPy array
    combinations_array = np.array(combinations)

    # Reshape the array to the correct dimensions
    shape = [len(lst) for lst in input_list] + [len(input_list)]
    n_dim_array = combinations_array.reshape(shape)
    return n_dim_array

#function that takes a lists of lists and generates an array of all possible combinations of entries in the lists
#e.g: [[1,2],[3,4]] -> [[1, 3],[1, 4],[2, 3],[2, 4]] (shape=4,2)
def gen_all_comb_from_list_of_lists_2(input_list):
    #generate all possible combinations using itertools.product
    combinations = list(itertools.product(*input_list))

    #convert the combinations to a NumPy array
    combinations_array = np.array(combinations)

    return combinations_array
    
    
    
##########
#Functions for Bayesian optimization using Gaussian process
#Partly based on code from lecture given by Prof. Pascal Friederich at KIT
########## 
  
def kernel(a, b):
    sqdist = np.sum(a**2, 1).reshape(-1, 1)       + np.sum(b**2, 1) - 2*np.dot(a, b.T)
    return(np.exp(-0.5*sqdist))

class gp:
    def __init__(self, kernel):
        self.kernel = kernel
        
    def train(self, X, f):
        self.X = X
        self.f = f
        self.K = self.kernel(self.X, self.X)
        self.Kinv = np.linalg.inv(self.K + 1e-6*np.eye((len(X))))
        
    def predict(self, X_pred):
        Ks = self.kernel(self.X, X_pred)
        Kss = self.kernel(X_pred, X_pred)
        mu_s = Ks.T.dot(self.Kinv).dot(self.f)
        cov_s = Kss - Ks.T.dot(self.Kinv).dot(Ks)
        mu = mu_s.ravel()
        uncertainty = np.sqrt(np.diag(cov_s))
        return(mu, uncertainty)

# Acquisition function
def acquisition_function_upper_confidence_bound(mu, uncertainty):
    ac = mu + uncertainty
    return(ac)

# Acquisition function
def acquisition_function_upper_confidence_bound_mod(mu, uncertainty,a,b):
    ac = mu*a + uncertainty*b
    return(ac)

#https://medium.com/@okanyenigun/step-by-step-guide-to-bayesian-optimization-a-python-based-approach-3558985c6818

def acquisition_function_expected_improvement(mu, uncertainty, best_f):
    y_pred, y_std = mu, uncertainty
    z = (y_pred - best_f) / y_std
    ei = (y_pred - best_f) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

def acquisition_function_probability_of_improvement(mu, uncertainty, best_f):
    y_pred, y_std = mu, uncertainty
    z = (y_pred - best_f) / y_std
    pi = norm.cdf(z)
    return pi    






##########
#Functions for angle evaluation of readdy sim
##########

def angle_1(v1,v2):
    v1_n=v1/np.linalg.norm(v1)
    v2_n=v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_n,v2_n),-1,1))


def cent_of_mass(list_of_coords):
    #list_of_coords= [[x1,y1,z1], [x2,y2,z2],...]
    x, y, z = np.array(list_of_coords).T
    cent = [np.mean(x), np.mean(y), np.mean(z)]
    return np.asarray(cent)
#angle between vectors in arms defined by tip and base of particles in each arm
def conf_angles_r_1(pos,types,traj,arms_names):
    angles_arms=[]
    
    #pos of particles are in same order for each time step
    #get pos in array from first time step
    arms_names_pos=[]
    for g in range(len(arms_names)):
        arms_names_pos_e=[]
        arm_name_sel=arms_names[g][0]
        base_name_sel=arms_names[g][1]
        for s in range(len(pos[0])):
            if traj.species_name(types[0][s])==arm_name_sel:
                arms_names_pos_e.append(s)
                
        for t in range(len(pos[0])):
             if traj.species_name(types[0][t])==base_name_sel:
                arms_names_pos_e.append(t) 
                
        arms_names_pos.append(arms_names_pos_e) 
    #print(arms_names_pos)
    #get all vectors for time point i
    
    for i in range(len(pos)):#iterate time points
        
        #vectors at given time: 
        list_of_vectors=[] 
        for n in range(len(arms_names_pos)):#iterate arms
            pos_1=arms_names_pos[n][0]
            pos_2=arms_names_pos[n][1]
            
            list_of_vectors.append(pos[i][pos_1]-pos[i][pos_2])
            
        #angles at given time:   
        angles_arms_t=[]
        for z in range(len(list_of_vectors)):
            ind1=z
            ind2=(z+1)%len(list_of_vectors) #e.g. 0-1, 1-2, 2-0
            #print(ind1,ind2)
            angles_arms_t.append(angle_1(list_of_vectors[ind1],list_of_vectors[ind2])) 
            
        angles_arms.append(angles_arms_t)
    return np.asarray(angles_arms)

#get angles between vectors aligned to the arms, between opposing arms
def conf_angles_r_1_opp(pos,types,traj,arms_names):
    angles_arms=[]
    
    #pos of particles are in same order for each time step
    #get pos in array from first time step
    arms_names_pos=[]
    for g in range(len(arms_names)):
        arms_names_pos_e=[]
        arm_name_sel=arms_names[g][0]
        base_name_sel=arms_names[g][1]
        for s in range(len(pos[0])):
            if traj.species_name(types[0][s])==arm_name_sel:
                arms_names_pos_e.append(s)
                
        for t in range(len(pos[0])):
             if traj.species_name(types[0][t])==base_name_sel:
                arms_names_pos_e.append(t) 
                
        arms_names_pos.append(arms_names_pos_e) 
    #print(arms_names_pos)
    #get all vectors for time point i
    
    for i in range(len(pos)):#iterate time points
        
        #vectors at given time: 
        list_of_vectors=[] 
        for n in range(len(arms_names_pos)):#iterate arms
            pos_1=arms_names_pos[n][0]
            pos_2=arms_names_pos[n][1]
            
            list_of_vectors.append(pos[i][pos_1]-pos[i][pos_2])
            
        #angles at given time:   
        angles_arms_t=[]
        for z in range(int(len(list_of_vectors)/2)):
            ind1=z
            ind2=z+int(len(list_of_vectors)/2) #e.g. 0-2, 1-3
            #print(ind1,ind2)
            angles_arms_t.append(angle_1(list_of_vectors[ind1],list_of_vectors[ind2])) 
            
        angles_arms.append(angles_arms_t)
    return np.asarray(angles_arms)


#get angles between vectors defined by two particles for each arm e.g. for angle with sticky ends
def conf_angles_group_1(pos,types,traj,arms_names):
    angles_arms=[]
    #pos of particles are in same order for each time step
    #get pos in array from first time step
    arms_names_pos=[]
    for g in range(len(arms_names)):

        arms_names_pos_e=np.zeros(4)
        point_name_1=arms_names[g][0]
        base_name_1=arms_names[g][1]
        point_name_2=arms_names[g][2]
        base_name_2=arms_names[g][3]
        #print(point_name_1,base_name_1,point_name_2,base_name_2)
        #iterate over indices of positions
        #get indices for one arm
        for s in range(len(pos[0])):
            if traj.species_name(types[0][s])==point_name_1:
                arms_names_pos_e[0]=s
            
            if traj.species_name(types[0][s])==base_name_1:
                arms_names_pos_e[1]=s 
                
            if traj.species_name(types[0][s])==point_name_2:
                arms_names_pos_e[2]=s 
                
            if traj.species_name(types[0][s])==base_name_2:
                arms_names_pos_e[3]=s 
                
        #fill with lists indices for all arms
        arms_names_pos.append(arms_names_pos_e) 
    #print(arms_names_pos)

    for i in range(len(pos)):#iterate time points
        
        #vectors at given time: 
        angles_arms_t=[]
        for n in range(len(arms_names_pos)):#iterate arms
 
            pos_point_1=int(arms_names_pos[n][0])
            pos_base_1=int(arms_names_pos[n][1])
            pos_point_2=int(arms_names_pos[n][2])
            pos_base_2=int(arms_names_pos[n][3])
            
            vec_1=pos[i][pos_point_1]-pos[i][pos_base_1]
            vec_2=pos[i][pos_point_2]-pos[i][pos_base_2]
            #print(vec_1,vec_2,angle_1(vec_1,vec_2), np.dot(vec_1/np.linalg.norm(vec_1),vec_2/np.linalg.norm(vec_2)))
            angles_arms_t.append(angle_1(vec_1,vec_2))


        angles_arms.append(angles_arms_t)
    return np.asarray(angles_arms)

#based on conf_angles_group_1
#use monomers with unique names to select positions of non-unique monomers by close distance
#only works if the two fused arms have unique fused monnomers (e.g. y_link_fused_a_1 and y_link_fused_c_1)
def conf_angles_2_bound_x_motifs_1(pos,types,traj,arms_names):
    
    #arms_names=[y_arm_(1)_,y_link_fused_(1)_,y_link_fused_(2)_,y_arm_(1)]
    #arms names are not unique (i.e. y_arm_(1)_exists twice) 
    #get indices of y_link_fused_(1/2)_, wich have to be unique for this to work, to find the closer corresponding arm index
    
    angles_fused_arms=[]
  
    #get correct index for y_arm_(1/2)_
    
    possible_index_y_arm_1=[]
    possible_index_y_arm_2=[]
    for i in range(len(pos[0])): #check first time step for both y_arm_(1/2)_ positions
        print(traj.species_name(types[0][i]),"#")
        if traj.species_name(types[0][i])==arms_names[0]:
            #print("!!")
            possible_index_y_arm_1.append(int(i))
            
        if traj.species_name(types[0][i])==arms_names[3]:
            possible_index_y_arm_2.append(int(i))
        
        if traj.species_name(types[0][i])==arms_names[1]:
            index_y_link_fused_1=int(i)
            
        if traj.species_name(types[0][i])==arms_names[2]:
            index_y_link_fused_2=int(i)
    #print("index",possible_index_y_arm_1,possible_index_y_arm_2,index_y_link_fused_1,index_y_link_fused_2)
    distance_possible_index_y_arm_1=[]
    distance_possible_index_y_arm_2=[]
    for i in range(len(possible_index_y_arm_1)): #get distances from unique fused indeces to all non-uniqe arms
        distance_possible_index_y_arm_1.append( np.sqrt(np.sum( (pos[0][possible_index_y_arm_1[i]] - pos[0][index_y_link_fused_1])**2  ) ) )
        distance_possible_index_y_arm_2.append( np.sqrt(np.sum( (pos[0][possible_index_y_arm_2[i]] - pos[0][index_y_link_fused_2])**2  ) ) )
        
    #print("dist",distance_possible_index_y_arm_1,distance_possible_index_y_arm_2)                                           
    #get index of shorter distance
    index_shorter_dist_arm_1=np.argmin(distance_possible_index_y_arm_1)
    index_shorter_dist_arm_2=np.argmin(distance_possible_index_y_arm_2)
    #print("index_short dist",index_shorter_dist_arm_1,index_shorter_dist_arm_2)                                          
    index_y_arm_1=int(possible_index_y_arm_1[index_shorter_dist_arm_1])
    index_y_arm_2=int(possible_index_y_arm_2[index_shorter_dist_arm_2])                                
    #print("index",index_y_arm_1,index_y_arm_2)                                     
    for i in range(len(pos)):
        #get centre of mass of two y_link_fused monomers
        cent_of_mass_y_link_fused=cent_of_mass( list_of_coords=[pos[i][index_y_link_fused_1],pos[i][index_y_link_fused_2]] )
        #cent_of_mass_y_link_fused=pos[i][index_y_link_fused_1]
        vec_1=cent_of_mass_y_link_fused - pos[i][index_y_arm_1]
        vec_2=pos[i][index_y_arm_2] -cent_of_mass_y_link_fused
        angles_fused_arms.append(angle_1(vec_1,vec_2))
                                               
    return np.asarray(angles_fused_arms)

#get angle distributions for four angles (base, linkers, arm axis, oppsing arms), obtain full distribution before hist application 
def get_training_angle_distr_7(folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict):
    #angle base-core-base
    angle_1_distribution_all=[]
    #mean_1_all=[]
    #std_1_all=[]
    
    #angles with sticky ends
    angle_2_distribution_all=[]
    #mean_2_all=[]
    #std_2_all=[]
    
    #angle along arm axis
    angle_3_distribution_all=[]
    #mean_3_all=[]
    #std_3_all=[]
    
    #fig=plt.figure(figsize=(4,4))            

    #load trajectory
    input_name=folder+name #+".h5"

    traj=readdy.Trajectory(input_name)

    times,types,ids,part_positions=traj.read_observable_particles()

    #angle between base-core-base
    #calculate angles
    angles_arms_all_t_1=conf_angles_r_1(pos=part_positions[t_skip:t_lim:t_step],types=types[t_skip:t_lim:t_step],traj=traj,arms_names=arms_names_1)
    angles_arms_1f=angles_arms_all_t_1.flatten()

    #get angle distribution
    #plt.subplot(211)
    #plt.xlabel("Angle [deg]")
    #plt.ylabel("Normalized frequency")
    #angles_test_hist_1=plt.hist(angles_arms_1f*180/np.pi,bins=np.arange(0,190,10),density=True, histtype='step')[0]
    #angle_1_distribution_all.append(angles_test_hist_1)

    #get mean, std
    #mean_1_all.append(np.mean(angles_arms_1f*180/np.pi))
    #std_1_all.append(np.std(angles_arms_1f*180/np.pi) )

    ##########
    ##########

    #angle with sticky ends
    angles_arms_all_t_2=conf_angles_group_1(pos=part_positions[t_skip:t_lim:t_step],types=types[t_skip:t_lim:t_step],traj=traj,arms_names=arms_names_2)
    angles_arms_2f=angles_arms_all_t_2.flatten()         

    #get angle distribution
    #plt.subplot(211)
    #plt.xlabel("Angle [deg]")
    #plt.ylabel("Normalized frequency")
    #angles_test_hist_2=plt.hist(angles_arms_2f*180/np.pi,bins=np.arange(0,190,10),density=True, histtype='step',ls="--")[0]
    #angle_2_distribution_all.append(angles_test_hist_2)

    #get mean, std
    #mean_2_all.append(np.mean(angles_arms_2f*180/np.pi))
    #std_2_all.append(np.std(angles_arms_2f*180/np.pi) )
    
    #########
    #########
    #angle along arm axis
    angles_arms_all_t_3=conf_angles_group_1(pos=part_positions[t_skip:t_lim:t_step],types=types[t_skip:t_lim:t_step],traj=traj,arms_names=arms_names_3)
    angles_arms_3f=angles_arms_all_t_3.flatten()         

    #get angle distribution
    #plt.subplot(211)
    #plt.xlabel("Angle [deg]")
    #plt.ylabel("Normalized frequency")
    #angles_test_hist_3=plt.hist(angles_arms_2f*180/np.pi,bins=np.arange(0,190,1),density=True, histtype='step',ls="--")[0]
    #angle_3_distribution_all.append(angles_test_hist_3)

    #get mean, std
    #mean_3_all.append(np.mean(angles_arms_3f*180/np.pi))
    #std_3_all.append(np.std(angles_arms_3f*180/np.pi) )


    #np.asarray(angles_arms_all_t_1), np.asarray(mean_1_all), np.asarray(std_1_all) ,np.asarray(angles_arms_all_t_2), np.asarray(mean_2_all), np.asarray(std_2_all)
    #return np.asarray(angles_arms_all_t_1) ,np.asarray(angles_arms_all_t_2)
    key_name=name  #+".h5"
    return_dict[key_name]=[np.asarray(angles_arms_all_t_1) ,np.asarray(angles_arms_all_t_2),np.asarray(angles_arms_all_t_3)] #,np.asarray(angles_arms_all_t_4)]

#get angle distributions for two bound x-motifs, obtain full distribution before hist application 
def get_training_angle_distr_9(folder,name,arms_names_2,t_skip,t_lim,t_step,return_dict):

    
    #angles with sticky ends
    angle_2_distribution_all=[]
    mean_2_all=[]
    std_2_all=[]
    
    #fig=plt.figure(figsize=(4,4))            

    #load trajectory
    input_name=folder+name #+".h5"

    traj=readdy.Trajectory(input_name)

    times,types,ids,part_positions=traj.read_observable_particles()


    #angle with sticky ends
    angles_arms_all_t_2=conf_angles_2_bound_x_motifs_1(pos=part_positions[t_skip:t_lim:t_step],types=types[t_skip:t_lim:t_step],traj=traj,arms_names=arms_names_2)
    angles_arms_2f=angles_arms_all_t_2.flatten()         

    #get angle distribution
    #plt.subplot(211)
    #plt.xlabel("Angle [deg]")
    #plt.ylabel("Normalized frequency")
    #angles_test_hist_2=plt.hist(angles_arms_2f*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #angle_2_distribution_all.append(angles_test_hist_2)

    #get mean, std
    #mean_2_all.append(np.mean(angles_arms_2f*180/np.pi))
    #std_2_all.append(np.std(angles_arms_2f*180/np.pi) )

    #np.asarray(angles_arms_all_t_1), np.asarray(mean_1_all), np.asarray(std_1_all) ,np.asarray(angles_arms_all_t_2), np.asarray(mean_2_all), np.asarray(std_2_all)
    #return np.asarray(angles_arms_all_t_1) ,np.asarray(angles_arms_all_t_2)
    key_name=name #+".h5"
    return_dict[key_name]=[np.asarray(angles_arms_all_t_2)]



##########
#Functions for getting fitness score from based on angle distributions
##########

##ground truth functions
#gt function for one angles
def gt_1_1(distr_1  ,  distr_target_1):
    dev=np.sum(np.abs(distr_1-distr_target_1))
    return 1/(1+dev)

#gt function for two angles
def gt_1_2(distr_1,distr_2  ,  distr_target_1,distr_target_2):
    dev=np.sum(np.abs(distr_1-distr_target_1)) +np.sum(np.abs(distr_2-distr_target_2) )
    return 1/(1+dev)

#gt function for three angles
def gt_1_3(distr_1,distr_2,distr_3  ,  distr_target_1,distr_target_2,distr_target_3):
    dev=np.sum(np.abs(distr_1-distr_target_1)) +np.sum(np.abs(distr_2-distr_target_2)) +np.sum(np.abs(distr_3-distr_target_3) )
    return 1/(1+dev)
    
#gt function for four angles
def gt_1_4(distr_1,distr_2,distr_3,distr_4  ,  distr_target_1,distr_target_2,distr_target_3,distr_target_4):
    dev=np.sum(np.abs(distr_1-distr_target_1)) +np.sum(np.abs(distr_2-distr_target_2)) +np.sum(np.abs(distr_3-distr_target_3))+np.sum(np.abs(distr_4-distr_target_4) )
    return 1/(1+dev)


############
#Angle evaluation
############

#evaluate n sims in parallel, get angle distributions and apply fitnes function
#return a fitness score based on base_angle_distr
def eval_sim_1_1_ba(folder,names_sim_repeat,t_skip,t_lim,t_step,base_angle_distr_target):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)    
        
    #concatenate results from simulation repeats
    values_angle_base=[]
    #values_angle_link=[]
    #values_angle_arm_axis=[]
    for s in range(len(names_sim_repeat)):
        values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        #values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        #values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        values_angle_base.append(values_angle_base_e)
        #values_angle_link.append(values_angle_link_e)
        #values_angle_arm_axis.append(values_angle_arm_axis_e)

    values_angle_base=np.asarray(values_angle_base).flatten()
    #values_angle_link=np.asarray(values_angle_link).flatten()
    #values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten()  
    
    #get value of ground truth at parameter set evaluated
    base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]
    
    base_angle_distr=np.asarray(base_angle_distr)
    #link_angle_distr=np.asarray(link_angle_distr)
    #arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    
    #calculate fitness score
    gt_value=gt_1_1(base_angle_distr,base_angle_distr_target)
    
    return gt_value

#evaluate n sims in parallel, get angle distributions and apply fitnes function
#return a fitness score based on link_angle_distr
def eval_sim_1_1_la(folder,names_sim_repeat,t_skip,t_lim,t_step,link_angle_distr_target):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    
    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)    
        
    #concatenate results from simulation repeats
    #values_angle_base=[]
    values_angle_link=[]
    #values_angle_arm_axis=[]
    for s in range(len(names_sim_repeat)):
        #values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        #values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        #values_angle_base.append(values_angle_base_e)
        values_angle_link.append(values_angle_link_e)
        #values_angle_arm_axis.append(values_angle_arm_axis_e)

    #values_angle_base=np.asarray(values_angle_base).flatten()
    values_angle_link=np.asarray(values_angle_link).flatten()
    #values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten()  
    
    #get value of ground truth at parameter set evaluated
    #base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]
    
    #base_angle_distr=np.asarray(base_angle_distr)
    link_angle_distr=np.asarray(link_angle_distr)
    #arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    
    #calculate fitness score
    gt_value=gt_1_1(link_angle_distr,link_angle_distr_target)
    
    return gt_value

#evaluate n sims in parallel, get angle distributions and apply fitnes function
#return a fitness score based on link_angle_distr
def eval_sim_1_1_aa(folder,names_sim_repeat,t_skip,t_lim,t_step,arm_axis_angle_distr_target):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)    
        
    #concatenate results from simulation repeats
    #values_angle_base=[]
    #values_angle_link=[]
    values_angle_arm_axis=[]
    for s in range(len(names_sim_repeat)):
        #values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        #values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        #values_angle_base.append(values_angle_base_e)
        #values_angle_link.append(values_angle_link_e)
        values_angle_arm_axis.append(values_angle_arm_axis_e)

    #values_angle_base=np.asarray(values_angle_base).flatten()
    #values_angle_link=np.asarray(values_angle_link).flatten()
    values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten() 
    
    #get value of ground truth at parameter set evaluated
    #base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]
    
    #base_angle_distr=np.asarray(base_angle_distr)
    #link_angle_distr=np.asarray(link_angle_distr)
    arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    #calculate fitness score
    gt_value=gt_1_1(arm_axis_angle_distr,arm_axis_angle_distr_target)
    
    return gt_value

#evaluate n sims in parallel, get angle distributions and apply fitnes function
#return a fitness score based on link_HP_angle_distr
def eval_sim_1_1_hp(folder,names_sim_repeat,t_skip,t_lim,t_step,link_HP_angle_distr_target):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link HP angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_hp_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_hp_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_hp_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)    
        
    #concatenate results from simulation repeats
    #values_angle_base=[]
    values_angle_link=[]
    #values_angle_arm_axis=[]
    for s in range(len(names_sim_repeat)):
        #values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        #values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        #values_angle_base.append(values_angle_base_e)
        values_angle_link.append(values_angle_link_e)
        #values_angle_arm_axis.append(values_angle_arm_axis_e)

    #values_angle_base=np.asarray(values_angle_base).flatten()
    values_angle_link=np.asarray(values_angle_link).flatten()
    #values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten() 
    
    #get value of ground truth at parameter set evaluated
    #base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]
    
    #base_angle_distr=np.asarray(base_angle_distr)
    link_angle_distr=np.asarray(link_angle_distr)
    #arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    
    #calculate fitness score
    gt_value=gt_1_1(link_angle_distr,link_HP_angle_distr_target)
    
    return gt_value
    
#evaluate n sims in parallel, get angle distributions and apply fitnes function
#return a fitness score based on angle_ds_fused_link_distr
def eval_sim_1_1_bm(folder,names_sim_repeat,t_skip,t_lim,t_step,angle_ds_fused_link_distr_target):
    #names_sim_repeat paths to n sims that are to be evaluated

    #arms_names_2_fused=["y_arm_a_1","y_link_fused_a_1","y_link_fused_b_1","y_arm_b_1"]
    #arms_names_2_fused=["y_arm_a_1","y_link_fused_a_1","y_link_fused_c_1","y_arm_c_1"]
    arms_names_2_fused=["y_arm_b_1","y_link_fused_b_1","y_link_fused_c_1","y_arm_c_1"]

    #arms_names_2_fused=["y_arm_a_1","y_link_fused_a_1","y_link_fused_a_1","y_arm_a_1"]
    #arms_names_2_fused=["y_arm_b_1","y_link_fused_b_1","y_link_fused_b_1","y_arm_b_1"]
    #arms_names_2_fused=["y_arm_c_1","y_link_fused_c_1","y_link_fused_c_1","y_arm_c_1"]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_2_fused,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_9)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)    
        

    values_angle_ds_fused_link=[]
    for s in range(len(names_sim_repeat)):

        
        values_angle_ds_fused_link_e=eval_dict[names_sim_repeat[s]][0].flatten()

        values_angle_ds_fused_link.append(values_angle_ds_fused_link_e)

    values_angle_ds_fused_link=np.asarray(values_angle_ds_fused_link)

    angle_ds_fused_link_distr=np.histogram(values_angle_ds_fused_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    

    angle_ds_fused_link_distr=np.asarray(angle_ds_fused_link_distr)
    
    
    #calculate fitness score
    gt_value=gt_1_1(angle_ds_fused_link_distr,angle_ds_fused_link_distr_target)
    
    return gt_value


#evaluate n sims in parallel, get angle distributions 
#return link_angle_distr,    values_angle_link
def cg_angle_distr_1_1_hp(folder,names_sim_repeat,t_skip,t_lim,t_step):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_hp_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_hp_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_hp_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)
    #concatenate results from simulation repeats
    values_angle_base=[]
    values_angle_link=[]
    values_angle_arm_axis=[]
    for s in range(len(names_sim_repeat)):
        values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        values_angle_base.append(values_angle_base_e)
        values_angle_link.append(values_angle_link_e)
        values_angle_arm_axis.append(values_angle_arm_axis_e)

    values_angle_base=np.asarray(values_angle_base).flatten()
    values_angle_link=np.asarray(values_angle_link).flatten()
    values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten()  
    
    #get value of ground truth at parameter set evaluated
    #base_angle_distr=plt.hist(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #link_angle_distr=plt.hist(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #arm_axis_angle_distr=plt.hist(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True, histtype='step',ls="--")[0]
    
    base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]

    
    base_angle_distr=np.asarray(base_angle_distr)
    link_angle_distr=np.asarray(link_angle_distr)
    arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    #calculate fitness score
    #gt_value=gt_1_4(base_angle_distr,arm_axis_angle_distr,link_angle_distr,base_angle_distr_target,arm_axis_angle_distr_target,link_angle_distr_target)
    
    return link_angle_distr, values_angle_link

#evaluate n sims in parallel, get angle distributions 
#angle_ds_fused_link_distr, values_angle_ds_fused_link
def cg_angle_distr_1_1_bm(folder,names_sim_repeat,t_skip,t_lim,t_step):
    #names_sim_repeat paths to n sims that are to be evaluated

    #arms_names_2_fused=["y_arm_a_1","y_link_fused_a_1","y_link_fused_b_1","y_arm_b_1"]
    #arms_names_2_fused=["y_arm_a_1","y_link_fused_a_1","y_link_fused_c_1","y_arm_c_1"]
    arms_names_2_fused=["y_arm_b_1","y_link_fused_b_1","y_link_fused_c_1","y_arm_c_1"]

    #arms_names_2_fused=["y_arm_a_1","y_link_fused_a_1","y_link_fused_a_1","y_arm_a_1"]
    #arms_names_2_fused=["y_arm_b_1","y_link_fused_b_1","y_link_fused_b_1","y_arm_b_1"]
    #arms_names_2_fused=["y_arm_c_1","y_link_fused_c_1","y_link_fused_c_1","y_arm_c_1"]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_2_fused,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_9)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)    
        

    values_angle_ds_fused_link=[]
    for s in range(len(names_sim_repeat)):

        
        values_angle_ds_fused_link_e=eval_dict[names_sim_repeat[s]][0].flatten()

        values_angle_ds_fused_link.append(values_angle_ds_fused_link_e)

    values_angle_ds_fused_link=np.asarray(values_angle_ds_fused_link).flatten()

    angle_ds_fused_link_distr=np.histogram(values_angle_ds_fused_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    

    angle_ds_fused_link_distr=np.asarray(angle_ds_fused_link_distr)
    
    
    #calculate fitness score
    #gt_value=gt_1_1(angle_ds_fused_link_distr,angle_ds_fused_link_distr_target)
    
    return angle_ds_fused_link_distr, values_angle_ds_fused_link

#Base Angle
def cg_angle_distr_1_1_ba(folder,names_sim_repeat,t_skip,t_lim,t_step):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)
    #concatenate results from simulation repeats
    values_angle_base=[]
    #values_angle_link=[]
    #values_angle_arm_axis=[]

    for s in range(len(names_sim_repeat)):
        values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        #values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        #values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        values_angle_base.append(values_angle_base_e)
        #values_angle_link.append(values_angle_link_e)
        #values_angle_arm_axis.append(values_angle_arm_axis_e)

    values_angle_base=np.asarray(values_angle_base).flatten()
    #values_angle_link=np.asarray(values_angle_link).flatten()
    #values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten() 
    
    #get value of ground truth at parameter set evaluated
    #base_angle_distr=plt.hist(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #link_angle_distr=plt.hist(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #arm_axis_angle_distr=plt.hist(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True, histtype='step',ls="--")[0]
    
    base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]

    
    base_angle_distr=np.asarray(base_angle_distr)
    #link_angle_distr=np.asarray(link_angle_distr)
    #arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    #calculate fitness score
    #gt_value=gt_1_4(base_angle_distr,arm_axis_angle_distr,link_angle_distr,      base_angle_distr_target,arm_axis_angle_distr_target_target,link_angle_distr_target)
    
    return base_angle_distr, values_angle_base


#Arm axis Angle
def cg_angle_distr_1_1_aa(folder,names_sim_repeat,t_skip,t_lim,t_step):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)
    #concatenate results from simulation repeats
    #values_angle_base=[]
    #values_angle_link=[]
    values_angle_arm_axis=[]

    for s in range(len(names_sim_repeat)):
        #values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        #values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        #values_angle_base.append(values_angle_base_e)
        #values_angle_link.append(values_angle_link_e)
        values_angle_arm_axis.append(values_angle_arm_axis_e)

    #values_angle_base=np.asarray(values_angle_base).flatten()
    #values_angle_link=np.asarray(values_angle_link).flatten()
    values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten() 
    
    #get value of ground truth at parameter set evaluated
    #base_angle_distr=plt.hist(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #link_angle_distr=plt.hist(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #arm_axis_angle_distr=plt.hist(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True, histtype='step',ls="--")[0]
    
    #base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    #link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]

    
    #base_angle_distr=np.asarray(base_angle_distr)
    #link_angle_distr=np.asarray(link_angle_distr)
    arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    #calculate fitness score
    #gt_value=gt_1_4(base_angle_distr,arm_axis_angle_distr,link_angle_distr,      base_angle_distr_target,arm_axis_angle_distr_target_target,link_angle_distr_target)
    
    return arm_axis_angle_distr, values_angle_arm_axis
    
def cg_angle_distr_1_3_ba_aa_la(folder,names_sim_repeat,t_skip,t_lim,t_step):
    #names_sim_repeat paths to n sims that are to be evaluated
    
    #need as target to define: base_angle_distr,link_angle_distr,arm_axis_angle_distr
    #define names of particles for angles
    #names base angle 
    #get names of particles for vectors for angles, first entry: name of particle further away from centre
    arms_names_1=[["y_arm_a_1","y_base_arm_a_1"],
                  ["y_arm_b_1","y_base_arm_b_1"],
                  ["y_arm_c_1","y_base_arm_c_1"]]

    #names link angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_2=[["y_link_a_1","y_arm_a_1","y_arm_a_1","y_base_arm_a_1"],
                  ["y_link_b_1","y_arm_b_1","y_arm_b_1","y_base_arm_b_1"],
                  ["y_link_c_1","y_arm_c_1","y_arm_c_1","y_base_arm_c_1"]]

    #names arm axis angle 
    #each entry contains name of tip, then base of first vector and then tip and base of second vector, then repeat for next arm
    arms_names_3=[["y_arm_a_1","y_base_arm_a_1","y_base_arm_a_1","y_centre_1"],
                  ["y_arm_b_1","y_base_arm_b_1","y_base_arm_b_1","y_centre_1"],
                  ["y_arm_c_1","y_base_arm_c_1","y_base_arm_c_1","y_centre_1"]]

    #set up parallel evaluation of angles
    f_eval_calls=[] #list of functions for evaluation in parallel
    input_params=[] #list of lists of parameters to use in each evaluation
    for j in range(len(names_sim_repeat)):
        #folder,name,arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step,return_dict
        input_params_e=[folder,names_sim_repeat[j],arms_names_1,arms_names_2,arms_names_3,t_skip,t_lim,t_step]
        
        f_eval_calls.append(get_training_angle_distr_7)
        input_params.append(input_params_e)
        
    #run len(names_sim_repeat) of evaluation for each sim in parallel
    eval_dict=run_p2(f_eval_calls,input_params)
    #concatenate results from simulation repeats
    values_angle_base=[]
    values_angle_link=[]
    values_angle_arm_axis=[]
    for s in range(len(names_sim_repeat)):
        values_angle_base_e=eval_dict[names_sim_repeat[s]][0].flatten()
        values_angle_link_e=eval_dict[names_sim_repeat[s]][1].flatten()
        values_angle_arm_axis_e=eval_dict[names_sim_repeat[s]][2].flatten()

        values_angle_base.append(values_angle_base_e)
        values_angle_link.append(values_angle_link_e)
        values_angle_arm_axis.append(values_angle_arm_axis_e)

    values_angle_base=np.asarray(values_angle_base).flatten()
    values_angle_link=np.asarray(values_angle_link).flatten()
    values_angle_arm_axis=np.asarray(values_angle_arm_axis).flatten() 
    
    #get value of ground truth at parameter set evaluated
    #base_angle_distr=plt.hist(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #link_angle_distr=plt.hist(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True, histtype='step',ls="--")[0]
    #arm_axis_angle_distr=plt.hist(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True, histtype='step',ls="--")[0]
    
    base_angle_distr=np.histogram(values_angle_base*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    link_angle_distr=np.histogram(values_angle_link*180/np.pi,bins=np.arange(0,190,5),density=True)[0]
    arm_axis_angle_distr=np.histogram(values_angle_arm_axis*180/np.pi,bins=np.arange(0,190,1),density=True)[0]

    
    base_angle_distr=np.asarray(base_angle_distr)
    link_angle_distr=np.asarray(link_angle_distr)
    arm_axis_angle_distr=np.asarray(arm_axis_angle_distr)
    
    #calculate fitness score
    #gt_value=gt_1_4(base_angle_distr,arm_axis_angle_distr,link_angle_distr,base_angle_distr_target,arm_axis_angle_distr_target,link_angle_distr_target)
    
    return base_angle_distr,link_angle_distr,arm_axis_angle_distr, values_angle_base,values_angle_link,values_angle_arm_axis



##########
#Functions for applying GP model and predicting next simulation parameters 
##########

#initialize GP model, train on input and predict next params
def apply_gp(x_rs,x_train_rs,x,f_train_rs,ac_function,ucb_norm_a,ucb_norm_b,repeat_measurements):
    #x_rs: full rescaled parameter space
    #x_train_rs: rescaled training values
    #x: full non-rescaled parameter space
    #initialize and train GP model
    gpr = gp(kernel)
    gpr.train(np.asarray(x_train_rs), np.asarray(f_train_rs))
    #print(x,x.shape,type(x))
    f_mean_pred, f_std_pred = gpr.predict(x_rs)

    #update model with new data point
    #upper conf bound
    if ac_function=="ucb":
        ac = acquisition_function_upper_confidence_bound_mod(f_mean_pred, f_std_pred,ucb_norm_a,ucb_norm_b)

    #expected improvement
    if ac_function=="ei":
        ac=acquisition_function_expected_improvement(mu=f_mean_pred, uncertainty=f_std_pred,best_f=np.max(f_train_rs))

    #prob of improvement
    if ac_function=="pi":
        ac=acquisition_function_probability_of_improvement(mu=f_mean_pred, uncertainty=f_std_pred,best_f=np.max(f_train_rs))

    #next sim parameters
    #algorithm is allowed to repeat measurement at parameter set explored before
    if repeat_measurements==True:
        id_new_sim=np.argmax(ac)   

        #scaled x values for fitting
        x_rs_next = np.asarray(x_rs[id_new_sim])
        #print("id_new_sim:",id_new_sim,x_next)
        #x_train.append(x_next)

        #unscaled x values for sim parameters
        x_next=np.asarray(x[id_new_sim])
        #print("new sim params:",x_sim_values_next)

    #next best set of parameters, not used before
    if repeat_measurements==False:
        #get indices sorted by value of ac function
        ids_sorted = sorted(range(len(ac)), key=lambda i: ac[i], reverse=True)

        #get first index not used before for next simulation
        for k in range(len(ids_sorted)):
            id_test=ids_sorted[k]
            x_test=x_rs[id_test]
            #special case: need initial id_new_sim if x only has one entry due to constrained dimension, can't be random sample to allow loading of existing sims:
            id_new_sim=0
            #print("id_test: ", type(id_test),"x test: ", list(x_test),"x_test type: ",type(list(x_test)),"x_train: ",type(x_train) )
            if list(x_test) not in np.asarray(x_train_rs).tolist():
            #if any(x_test == s for s in x_train):  
                id_new_sim=id_test
                x_rs_next=np.asarray(x_test)
                break


        #scaled x values for fitting
        x_rs_next = np.asarray(x_rs[id_new_sim])
        #print("id_new_sim:",id_new_sim,x_next)
        #x_train.append(x_next)

        #unscaled x values for sim parameters
        x_next=np.asarray(x[id_new_sim])
        #print("new sim params:",x_sim_values_next)
    return x_rs_next, x_next, gpr


##########
#Run Bayesian optimization
##########

#this function can be used to run Bayesian opt on any type of sim, as long as the input format is compatible 
def run_gp_10(df_run_sim,
              df_var_params,
              fct_sim,
              df_eval_opt_param_set,
              sim_folder,
              sim_name,
              file_name_sim_params,
              sim_repeats,
              iterations,
              print_sim_config,
              load_existing_init_sims,
              load_existing_opt_sims):
    
    #df_run_sim: dataframe of parameters needed in simulation
    #df_var_params: dataframe of subset of paramters that are to be optimized
    #fct_sim: function that runs simulation with parameters from df_run_sim and df_var_params
    #df_eval_opt_param_set: dataframe of settings for evaluating, and bayesian opt for each independent parameter set
    #sim_folder: folder in which to store sim runs
    #file_name_sim_params: name of txt file for sim parameters, if None, no file saved
    #sim_repeats: how many repeats of each sim to use for averaging
    #iterations: how many interations of Bayesian opt to run
    #print_sim_config whether to supress printing of simulation parameters
    #load_existing_init_sims: option to load existing initial simulations in folder
    #load_existing_opt_sims: option to load existing optimization simulations in folder
    
    #generate a dict with key: set of indep params value: input matrix of all possible parameter comb.
    #parameter input matrix of form: m_set_1[param 1][param 2]...[param n]
    #do for params and rescaled params
    
    #update sim folder and sim name
    df_run_sim["name"]=sim_name
    df_run_sim["folder"]=sim_folder
    
    #group by 'Set of indep params' and create lists of 'Param values'
    df_var_params_gr = df_var_params.groupby('Set of indep params')['Param values'].apply(list).reset_index()
    df_var_rs_params_gr = df_var_params.groupby('Set of indep params')['Rs param values'].apply(list).reset_index()

    #convert to dictionary with lists of params.
    dict_var_params_gr = df_var_params_gr.set_index('Set of indep params')['Param values'].to_dict()
    dict_var_rs_params_gr = df_var_rs_params_gr.set_index('Set of indep params')['Rs param values'].to_dict()
    
    #convert to arrays of all possible param. combinations
    for key in dict_var_params_gr:
        #dict_var_params_gr[key]=gen_all_comb_from_list_of_lists(dict_var_params_gr[key])
        #dict_var_rs_params_gr[key]=gen_all_comb_from_list_of_lists(dict_var_rs_params_gr[key])
        #append to x and x_sim in df_eval_opt_param_set dataframe
        x_rs_var_params=gen_all_comb_from_list_of_lists_2(dict_var_rs_params_gr[key])
        x_var_params=gen_all_comb_from_list_of_lists_2(dict_var_params_gr[key])
        df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==key]["GP params"].values[0]["x_rs"]=x_rs_var_params
        df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==key]["GP params"].values[0]["x"]=x_var_params

    #create a dataframe with Set of indep param column, and list of all param names for later reference, which params are in wich set
    #the order of the names will be the same order of the parameters in updates provided by GP model 
    #example:Set of indep param | Param Name
    #        0                    k1,k3
    #        1                    k2 
    
    #group by 'Set of indep params' and create lists of 'Param names'
    df_var_params_gr_names = df_var_params.groupby('Set of indep params')['Param names'].apply(list).reset_index()
    
    #create entry for names of parameters in each independent set in df df_eval_opt_param_set
    df_eval_opt_param_set["Param names in set"]=None
    for index, row in df_var_params_gr_names.iterrows():
        param_set_sel=row["Set of indep params"]
        df_eval_opt_param_set.at[param_set_sel,"Param names in set"]=row['Param names']
        
    
    
    #does not work as storing n-dim arrays causes problems!!
    #add entry of parameter combinations to df_eval_opt_param_set df
    #df_var_params_comb=pd.DataFrame(dict_var_params_gr)
    #df_var_rs_params_comb=pd.DataFrame(dict_var_rs_params_gr)
    
    #df_eval_opt_param_set=pd.concat([df_eval_opt_param_set,df_var_params_comb], ignore_index=True)
    #df_eval_opt_param_set=pd.concat([df_eval_opt_param_set,df_var_rs_params_comb], ignore_index=True)
    
    #create columns for fitness,rs fitness, x train rs and x train for each indep param set
    df_eval_opt_param_set["f train"]=[[] for _ in range(len(df_eval_opt_param_set))]
    df_eval_opt_param_set["f train rs"]=[[] for _ in range(len(df_eval_opt_param_set))]

    df_eval_opt_param_set["x train rs"]=[[] for _ in range(len(df_eval_opt_param_set))]
    df_eval_opt_param_set["x train"]=[[] for _ in range(len(df_eval_opt_param_set))]
        
    #run simulations with initial parameter guesses before setting up GP models   
    #open file for writing sim paramters
    sim_count=0 #counter for all simulations (initial guesses and optimization run) for later identification by number
    sim_doc_file=[] #list with all the string documenation to save later
    #iterate over number of initial simulations
    for i in range(len(df_var_params["Index init guess"][0])):
        print("Step ", i+1,"/",len(df_var_params["Index init guess"][0])," initial simulations")
        #iterate over variable parameters
        
        #create dictionaries for chosen variable parameters with and without rs, at current iteration of initial sim
        dict_sel_param={}
        dict_sel_rs_param={}
        
        #when appending the chosen x train and x train rs values, entries in list of list must be split by simulation iteration
        #i.e.[[x train step 1], [x train step 2],...], therefore create new sub_df for each inital sim
        df_eval_opt_param_set_it=pd.DataFrame(df_eval_opt_param_set["Set of indep params"]) #create DF with Set of indep params as column
        df_eval_opt_param_set_it["x train rs it"]=[[] for _ in range(len(df_eval_opt_param_set_it))]
        df_eval_opt_param_set_it["x train it"]=[[] for _ in range(len(df_eval_opt_param_set_it))]
        
        for index, row in df_var_params.iterrows():
            sel_param=row["Param names"]
            sel_rs_param=sel_param+"_rs"
            sel_index_init_guess=row["Index init guess"][i] 
            sel_param_value=row["Param values"][sel_index_init_guess]
            sel_rs_param_value=row["Rs param values"][sel_index_init_guess]
            
            #set variable params
            dict_sel_param[sel_param]=sel_param_value
            dict_sel_rs_param[sel_rs_param]=sel_rs_param_value
            
            #the chosen paramters need to be added to the df df_eval_opt_param_set containing x train  and x sim values
            param_set_index=row["Set of indep params"]
            #find correct row to append x train , x train rs
            row_index = df_eval_opt_param_set_it[df_eval_opt_param_set_it['Set of indep params'] == param_set_index].index.item()
            df_eval_opt_param_set_it.at[row_index,"x train rs it"].append(sel_rs_param_value)
            df_eval_opt_param_set_it.at[row_index,"x train it"].append(sel_param_value)
        
        #append lists of x train, x train rs to list (of lists) for training
        for index, row in df_eval_opt_param_set.iterrows():
            param_set_index=row["Set of indep params"]
            row_index = df_eval_opt_param_set_it[df_eval_opt_param_set_it['Set of indep params'] == param_set_index].index.item()

            row["x train rs"].append(df_eval_opt_param_set_it.loc[row_index,"x train rs it"])
            row["x train"].append(df_eval_opt_param_set_it.loc[row_index,"x train it"])
                    
        #update parameter dictionary with selected variable parameters   
        df_run_sim.update(dict_sel_param)
        
        #set up sim_repeats number of parallel simulations with same parameters for averaging    
        f_calls_repeats_initial=[] # list of functions for simulation to run in parallel
        input_params_initial=[] # list of lists of parameters to use in each simulation
        names_initial=[] #list of names of repeat simulation for later evaluation of angle distributions
        
        for j in range(sim_repeats): #use sim_repeats repeats of each parameter set for averaging
            sim_count=sim_count+1 #only paramter that changes is the name
            add_num_1=sim_count 

            #update name with add_num of sim and add to simulation input
            df_run_sim["add_num_1"]=add_num_1
            f_calls_repeats_initial.append(fct_sim)
            input_params_initial.append(df_run_sim.copy())

            #add name of sim to list for later evaluation
            names_initial.append(sim_name+str(sim_count)+".h5")
            #add name of sim to doc file
            sim_doc_file.append(sim_name+str(sim_count)+".h5")
            
        #run simulations 
        if load_existing_init_sims==False:
            if print_sim_config==False:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_p_dict(f_calls_repeats_initial,input_params_initial)
            if print_sim_config==True:
                run_p_dict(f_calls_repeats_initial,input_params_initial)
            
        #add sim params to doc file
        sim_doc_file.append(str(dict_sel_param))
        sim_doc_file.append(str(dict_sel_rs_param))
        sim_doc_file.append("#####")
        
        #run evaluation of initial simulations

        #iterate over independent parameter sets, evaluate for each param set and add fitness score
        for index, row in df_eval_opt_param_set.iterrows():
            sel_set=row["Set of indep params"]
            sel_eval=row["Eval fct"] #function for evaluation
            sel_eval_input=row["Eval params"] # parameters for evaluation
            sel_scaler=row["Scaler"] #Scaler for fitness 

            #get ground truth
            sel_eval_input["folder"]=sim_folder #update folder
            sel_eval_input["names_sim_repeat"]=names_initial #update names
            gt_value=sel_eval(**sel_eval_input) #same sims are evaluated for independent parameter sets

            #append ground truth
            row["f train"].append(gt_value) #add to list 

            #get rescaled ground truth/fitness
            scaler_gt=sel_scaler #()
            f_train_rs= scaler_gt.fit_transform(np.asarray(row["f train"]).reshape(-1, 1)).flatten()
            
            #add rs ground truth
            #have to manually insert entire rescaled f train array, cant use row[""] syntax, have to use .at
            id_sel_set=df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==sel_set].index.values[0]
            df_eval_opt_param_set.at[id_sel_set,"f train rs"]=f_train_rs
            #ad f_train_rs
            row["GP params"]["f_train_rs"]=f_train_rs  
            
    #link GP params dictionary entry for x_train_rs with row containing x_train_rs
    #only needs to be linked once, dict will be updated when updating list in df
    #for f_train_rs use manual appending to both GP params dict and list in df
    for index, row in df_eval_opt_param_set.iterrows():
        row["GP params"]["x_train_rs"]=row["x train rs"]
        #row["GP params"]["f_train_rs"]=row["f train rs"]
        
    #run iterations of optimization 
    for i in range(iterations):
        print("Step ", i+1,"/",iterations," optimization simulations")
        #create dictionaries for chosen variable parameters with and without rs, at current iteration of optimization
        dict_sel_param_opt={}
        dict_sel_rs_param_opt={}
        
        #iterate independent set of parameters and gp models
        for index, row in df_eval_opt_param_set.iterrows():
            #parameter set index
            param_set_index=row["Set of indep params"]
            
            #print("x train rs",index,i,len(row["GP params"]["x_train_rs"]))
            #print("f train rs",index,i,len(row["GP params"]["f_train_rs"]))
            #application of gp model for selected parameter needs updates to x_sim_values, x_train
            #print(len(row["GP params"]["x_train"]),len(row["GP params"]["x_train"][0]))
            #print(row["GP params"])
            #print(row["x train rs"])
            #print("#####")
            #print(row["GP params"]["x_train_rs"])
            x_rs_next, x_next, gpr=row["GP"](**row["GP params"])
            #print("len x rs",x_rs_next)
            #row["GP params"]["x_train_rs"].append(x_rs_next.tolist())
            
            row["x train"].append(x_next.tolist())
            row["x train rs"].append(x_rs_next.tolist())
            #row["x train rs"].append("test")
            #print(index,row["x train rs"][-1])
            
            #print("updated st x train rs",index,i,len(row["x train"]))
            #print("updated x train rs",index,i,len(row["GP params"]["x_train_rs"]))
            #assign param names to new values
            list_param_names_sel_set=df_var_params_gr_names[df_var_params_gr_names["Set of indep params"]==index]["Param names"].values[0]

            for k in range(len(list_param_names_sel_set)):
                param_name_sel=list_param_names_sel_set[k]
                param_name_sel_rs=param_name_sel+"_rs"
                dict_sel_param_opt[param_name_sel]=x_next[k]
                dict_sel_rs_param_opt[param_name_sel_rs]=x_rs_next[k]
        
        
        #update parameter dictionary with selected optimized parameters   
        df_run_sim.update(dict_sel_param_opt)
        
        #set up sim_repeats number of parallel simulations with same parameters for averaging    
        f_calls_repeats_opt=[] # list of functions for simulation to run in parallel
        input_params_opt=[] # list of lists of parameters to use in each simulation
        names_opt=[] #list of names of repeat simulation for later evaluation of angle distributions
        
        for j in range(sim_repeats): #use sim_repeats repeats of each parameter set for averaging
            sim_count=sim_count+1 #only paramter that changes is the name
            add_num_1=sim_count 

            #update name with add_num of sim and add to simulation input
            df_run_sim["add_num_1"]=add_num_1
            f_calls_repeats_opt.append(fct_sim)
            input_params_opt.append(df_run_sim.copy())

            #add name of sim to list for later evaluation
            names_opt.append(sim_name+str(sim_count)+".h5")
            #add name of sim to doc file
            sim_doc_file.append(sim_name+str(sim_count)+".h5")
        #run simulations 
        if load_existing_opt_sims==False:
            if print_sim_config==False:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_p_dict(f_calls_repeats_opt,input_params_opt)
            if print_sim_config==True:
                run_p_dict(f_calls_repeats_opt,input_params_opt)
        
        #add sim params to doc file
        sim_doc_file.append(str(dict_sel_param_opt))
        sim_doc_file.append(str(dict_sel_rs_param_opt))
        sim_doc_file.append("#####")
        #run evaluation of simulations
        
        #iterate over independent parameter sets, evaluate for each param set and add fitness score
        for index, row in df_eval_opt_param_set.iterrows():
            sel_set=row["Set of indep params"]
            sel_eval=row["Eval fct"] #function for evaluation
            sel_eval_input=row["Eval params"] # parameters for evaluation
            sel_scaler=row["Scaler"] #Scaler for fitness 
            
            #get ground truth
            sel_eval_input["folder"]=sim_folder #update folder
            sel_eval_input["names_sim_repeat"]=names_opt #update names
            gt_value=sel_eval(**sel_eval_input) #same sims are evaluated for independent parameter sets

            #append ground truth
            row["f train"].append(gt_value) #add to list 

            #get rescaled ground truth/fitness
            scaler_gt=sel_scaler #()
            f_train_rs= scaler_gt.fit_transform(np.asarray(row["f train"]).reshape(-1, 1)).flatten()
            #print("updated f train",len(f_train_rs))
            #add rs ground truth
            #have to manually insert entire rescaled f train array, cant use row[""] syntax, have to use .at
            id_sel_set=df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==sel_set].index.values[0]
            df_eval_opt_param_set.at[id_sel_set,"f train rs"]=f_train_rs
            
            #add x_train and x_sim_values
            #row["GP params"]["x_train_rs"]=row["x train rs"]
            row["GP params"]["f_train_rs"]=f_train_rs 
            
    #save sim_doc_file
    if file_name_sim_params!=None:
        sim_doc_file_save_name=sim_folder+file_name_sim_params
        np.savetxt(sim_doc_file_save_name,sim_doc_file,delimiter=" ", fmt="%s")
    #for index, row in df_eval_opt_param_set.iterrows():
        #row["x train rs"].append("test2")
        #row["f train rs"].append("train test")
    return df_eval_opt_param_set, gpr
    


#this function can be used to run Bayesian opt on any type of sim, as long as the input format is compatible 
#based on run_gp_10, with the option to continue optimization by specifying a the total number of existing simulations (load_existing_opt_sims) to be read in, before new simulations are added
def run_gp_11(df_run_sim,
              df_var_params,
              fct_sim,
              df_eval_opt_param_set,
              sim_folder,
              sim_name,
              file_name_sim_params,
              sim_repeats,
              iterations,
              print_sim_config,
              load_existing_init_sims,
              load_existing_opt_sims):
    
    #df_run_sim: dataframe of parameters needed in simulation
    #df_var_params: dataframe of subset of paramters that are to be optimized
    #fct_sim: function that runs simulation with parameters from df_run_sim and df_var_params
    #df_eval_opt_param_set: dataframe of settings for evaluating, and bayesian opt for each independent parameter set
    #sim_folder: folder in which to store sim runs
    #file_name_sim_params: name of txt file for sim parameters, if None, no file saved
    #sim_repeats: how many repeats of each sim to use for averaging
    #iterations: how many interations of Bayesian opt to run
    #print_sim_config whether to supress printing of simulation parameters
    #load_existing_init_sims: option to load existing initial simulations in folder
    #load_existing_opt_sims: option to load existing optimization simulations in folder, if set to number n, n simulations will be loaded, afterwards new simulations will be added
    
    #generate a dict with key: set of indep params value: input matrix of all possible parameter comb.
    #parameter input matrix of form: m_set_1[param 1][param 2]...[param n]
    #do for params and rescaled params
    
    #update sim folder and sim name
    df_run_sim["name"]=sim_name
    df_run_sim["folder"]=sim_folder
    
    #group by 'Set of indep params' and create lists of 'Param values'
    df_var_params_gr = df_var_params.groupby('Set of indep params')['Param values'].apply(list).reset_index()
    df_var_rs_params_gr = df_var_params.groupby('Set of indep params')['Rs param values'].apply(list).reset_index()

    #convert to dictionary with lists of params.
    dict_var_params_gr = df_var_params_gr.set_index('Set of indep params')['Param values'].to_dict()
    dict_var_rs_params_gr = df_var_rs_params_gr.set_index('Set of indep params')['Rs param values'].to_dict()
    
    #convert to arrays of all possible param. combinations
    for key in dict_var_params_gr:
        #dict_var_params_gr[key]=gen_all_comb_from_list_of_lists(dict_var_params_gr[key])
        #dict_var_rs_params_gr[key]=gen_all_comb_from_list_of_lists(dict_var_rs_params_gr[key])
        #append to x and x_sim in df_eval_opt_param_set dataframe
        x_rs_var_params=gen_all_comb_from_list_of_lists_2(dict_var_rs_params_gr[key])
        x_var_params=gen_all_comb_from_list_of_lists_2(dict_var_params_gr[key])
        df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==key]["GP params"].values[0]["x_rs"]=x_rs_var_params
        df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==key]["GP params"].values[0]["x"]=x_var_params

    #create a dataframe with Set of indep param column, and list of all param names for later reference, which params are in wich set
    #the order of the names will be the same order of the parameters in updates provided by GP model 
    #example:Set of indep param | Param Name
    #        0                    k1,k3
    #        1                    k2 
    
    #group by 'Set of indep params' and create lists of 'Param names'
    df_var_params_gr_names = df_var_params.groupby('Set of indep params')['Param names'].apply(list).reset_index()
    
    #create entry for names of parameters in each independent set in df df_eval_opt_param_set
    df_eval_opt_param_set["Param names in set"]=None
    for index, row in df_var_params_gr_names.iterrows():
        param_set_sel=row["Set of indep params"]
        df_eval_opt_param_set.at[param_set_sel,"Param names in set"]=row['Param names']
        
    
    
    #does not work as storing n-dim arrays causes problems!!
    #add entry of parameter combinations to df_eval_opt_param_set df
    #df_var_params_comb=pd.DataFrame(dict_var_params_gr)
    #df_var_rs_params_comb=pd.DataFrame(dict_var_rs_params_gr)
    
    #df_eval_opt_param_set=pd.concat([df_eval_opt_param_set,df_var_params_comb], ignore_index=True)
    #df_eval_opt_param_set=pd.concat([df_eval_opt_param_set,df_var_rs_params_comb], ignore_index=True)
    
    #create columns for fitness,rs fitness, x train rs and x train for each indep param set
    df_eval_opt_param_set["f train"]=[[] for _ in range(len(df_eval_opt_param_set))]
    df_eval_opt_param_set["f train rs"]=[[] for _ in range(len(df_eval_opt_param_set))]

    df_eval_opt_param_set["x train rs"]=[[] for _ in range(len(df_eval_opt_param_set))]
    df_eval_opt_param_set["x train"]=[[] for _ in range(len(df_eval_opt_param_set))]
        
    #run simulations with initial parameter guesses before setting up GP models   
    #open file for writing sim paramters
    sim_count=0 #counter for all simulations (initial guesses and optimization run) for later identification by number
    sim_doc_file=[] #list with all the string documenation to save later
    #iterate over number of initial simulations
    for i in range(len(df_var_params["Index init guess"][0])):
        print("Step ", i+1,"/",len(df_var_params["Index init guess"][0])," initial simulations")
        #iterate over variable parameters
        
        #create dictionaries for chosen variable parameters with and without rs, at current iteration of initial sim
        dict_sel_param={}
        dict_sel_rs_param={}
        
        #when appending the chosen x train and x train rs values, entries in list of list must be split by simulation iteration
        #i.e.[[x train step 1], [x train step 2],...], therefore create new sub_df for each inital sim
        df_eval_opt_param_set_it=pd.DataFrame(df_eval_opt_param_set["Set of indep params"]) #create DF with Set of indep params as column
        df_eval_opt_param_set_it["x train rs it"]=[[] for _ in range(len(df_eval_opt_param_set_it))]
        df_eval_opt_param_set_it["x train it"]=[[] for _ in range(len(df_eval_opt_param_set_it))]
        
        for index, row in df_var_params.iterrows():
            sel_param=row["Param names"]
            sel_rs_param=sel_param+"_rs"
            sel_index_init_guess=row["Index init guess"][i] 
            sel_param_value=row["Param values"][sel_index_init_guess]
            sel_rs_param_value=row["Rs param values"][sel_index_init_guess]
            
            #set variable params
            dict_sel_param[sel_param]=sel_param_value
            dict_sel_rs_param[sel_rs_param]=sel_rs_param_value
            
            #the chosen paramters need to be added to the df df_eval_opt_param_set containing x train  and x sim values
            param_set_index=row["Set of indep params"]
            #find correct row to append x train , x train rs
            row_index = df_eval_opt_param_set_it[df_eval_opt_param_set_it['Set of indep params'] == param_set_index].index.item()
            df_eval_opt_param_set_it.at[row_index,"x train rs it"].append(sel_rs_param_value)
            df_eval_opt_param_set_it.at[row_index,"x train it"].append(sel_param_value)
        
        #append lists of x train, x train rs to list (of lists) for training
        for index, row in df_eval_opt_param_set.iterrows():
            param_set_index=row["Set of indep params"]
            row_index = df_eval_opt_param_set_it[df_eval_opt_param_set_it['Set of indep params'] == param_set_index].index.item()

            row["x train rs"].append(df_eval_opt_param_set_it.loc[row_index,"x train rs it"])
            row["x train"].append(df_eval_opt_param_set_it.loc[row_index,"x train it"])
                    
        #update parameter dictionary with selected variable parameters   
        df_run_sim.update(dict_sel_param)
        
        #set up sim_repeats number of parallel simulations with same parameters for averaging    
        f_calls_repeats_initial=[] # list of functions for simulation to run in parallel
        input_params_initial=[] # list of lists of parameters to use in each simulation
        names_initial=[] #list of names of repeat simulation for later evaluation of angle distributions
        
        for j in range(sim_repeats): #use sim_repeats repeats of each parameter set for averaging
            sim_count=sim_count+1 #only paramter that changes is the name
            add_num_1=sim_count 

            #update name with add_num of sim and add to simulation input
            df_run_sim["add_num_1"]=add_num_1
            f_calls_repeats_initial.append(fct_sim)
            input_params_initial.append(df_run_sim.copy())

            #add name of sim to list for later evaluation
            names_initial.append(sim_name+str(sim_count)+".h5")
            #add name of sim to doc file
            sim_doc_file.append(sim_name+str(sim_count)+".h5")
            
        #run simulations 
        if load_existing_init_sims==False:
            if print_sim_config==False:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_p_dict(f_calls_repeats_initial,input_params_initial)
            if print_sim_config==True:
                run_p_dict(f_calls_repeats_initial,input_params_initial)
            
        #add sim params to doc file
        sim_doc_file.append(str(dict_sel_param))
        sim_doc_file.append(str(dict_sel_rs_param))
        sim_doc_file.append("#####")
        
        #run evaluation of initial simulations

        #iterate over independent parameter sets, evaluate for each param set and add fitness score
        for index, row in df_eval_opt_param_set.iterrows():
            sel_set=row["Set of indep params"]
            sel_eval=row["Eval fct"] #function for evaluation
            sel_eval_input=row["Eval params"] # parameters for evaluation
            sel_scaler=row["Scaler"] #Scaler for fitness 

            #get ground truth
            sel_eval_input["folder"]=sim_folder #update folder
            sel_eval_input["names_sim_repeat"]=names_initial #update names
            gt_value=sel_eval(**sel_eval_input) #same sims are evaluated for independent parameter sets

            #append ground truth
            row["f train"].append(gt_value) #add to list 

            #get rescaled ground truth/fitness
            scaler_gt=sel_scaler #()
            f_train_rs= scaler_gt.fit_transform(np.asarray(row["f train"]).reshape(-1, 1)).flatten()
            
            #add rs ground truth
            #have to manually insert entire rescaled f train array, cant use row[""] syntax, have to use .at
            id_sel_set=df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==sel_set].index.values[0]
            df_eval_opt_param_set.at[id_sel_set,"f train rs"]=f_train_rs
            #ad f_train_rs
            row["GP params"]["f_train_rs"]=f_train_rs  
            
    #link GP params dictionary entry for x_train_rs with row containing x_train_rs
    #only needs to be linked once, dict will be updated when updating list in df
    #for f_train_rs use manual appending to both GP params dict and list in df
    for index, row in df_eval_opt_param_set.iterrows():
        row["GP params"]["x_train_rs"]=row["x train rs"]
        #row["GP params"]["f_train_rs"]=row["f train rs"]
        
    #run iterations of optimization 
    for i in range(iterations):
        print("Step ", i+1,"/",iterations," optimization simulations")
        #create dictionaries for chosen variable parameters with and without rs, at current iteration of optimization
        dict_sel_param_opt={}
        dict_sel_rs_param_opt={}
        
        #iterate independent set of parameters and gp models
        for index, row in df_eval_opt_param_set.iterrows():
            #parameter set index
            param_set_index=row["Set of indep params"]
            
            #print("x train rs",index,i,len(row["GP params"]["x_train_rs"]))
            #print("f train rs",index,i,len(row["GP params"]["f_train_rs"]))
            #application of gp model for selected parameter needs updates to x_sim_values, x_train
            #print(len(row["GP params"]["x_train"]),len(row["GP params"]["x_train"][0]))
            #print(row["GP params"])
            #print(row["x train rs"])
            #print("#####")
            #print(row["GP params"]["x_train_rs"])
            x_rs_next, x_next, gpr=row["GP"](**row["GP params"])
            #print("len x rs",x_rs_next)
            #row["GP params"]["x_train_rs"].append(x_rs_next.tolist())
            
            row["x train"].append(x_next.tolist())
            row["x train rs"].append(x_rs_next.tolist())
            #row["x train rs"].append("test")
            #print(index,row["x train rs"][-1])
            
            #print("updated st x train rs",index,i,len(row["x train"]))
            #print("updated x train rs",index,i,len(row["GP params"]["x_train_rs"]))
            #assign param names to new values
            list_param_names_sel_set=df_var_params_gr_names[df_var_params_gr_names["Set of indep params"]==index]["Param names"].values[0]

            for k in range(len(list_param_names_sel_set)):
                param_name_sel=list_param_names_sel_set[k]
                param_name_sel_rs=param_name_sel+"_rs"
                dict_sel_param_opt[param_name_sel]=x_next[k]
                dict_sel_rs_param_opt[param_name_sel_rs]=x_rs_next[k]
        
        
        #update parameter dictionary with selected optimized parameters   
        df_run_sim.update(dict_sel_param_opt)
        
        #set up sim_repeats number of parallel simulations with same parameters for averaging    
        f_calls_repeats_opt=[] # list of functions for simulation to run in parallel
        input_params_opt=[] # list of lists of parameters to use in each simulation
        names_opt=[] #list of names of repeat simulation for later evaluation of angle distributions
        
        for j in range(sim_repeats): #use sim_repeats repeats of each parameter set for averaging
            sim_count=sim_count+1 #only paramter that changes is the name
            add_num_1=sim_count 

            #update name with add_num of sim and add to simulation input
            df_run_sim["add_num_1"]=add_num_1
            f_calls_repeats_opt.append(fct_sim)
            input_params_opt.append(df_run_sim.copy())

            #add name of sim to list for later evaluation
            names_opt.append(sim_name+str(sim_count)+".h5")
            #add name of sim to doc file
            sim_doc_file.append(sim_name+str(sim_count)+".h5")
        #run simulations 
        #no existing simulations to load
        if type(load_existing_opt_sims)!=int:
            if (load_existing_opt_sims==False):
                if print_sim_config==False:
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_p_dict(f_calls_repeats_opt,input_params_opt)
                if print_sim_config==True:
                    run_p_dict(f_calls_repeats_opt,input_params_opt)
   
        #load_existing_opt_sims is number of simulations to load, when sim_count > than existing number, do new simulations  
        if type(load_existing_opt_sims)==int:
            if (sim_count >load_existing_opt_sims):

                if print_sim_config==False:
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_p_dict(f_calls_repeats_opt,input_params_opt)
                if print_sim_config==True:
                    run_p_dict(f_calls_repeats_opt,input_params_opt)
                    
        #add sim params to doc file
        sim_doc_file.append(str(dict_sel_param_opt))
        sim_doc_file.append(str(dict_sel_rs_param_opt))
        sim_doc_file.append("#####")
        #run evaluation of simulations
        
        #iterate over independent parameter sets, evaluate for each param set and add fitness score
        for index, row in df_eval_opt_param_set.iterrows():
            sel_set=row["Set of indep params"]
            sel_eval=row["Eval fct"] #function for evaluation
            sel_eval_input=row["Eval params"] # parameters for evaluation
            sel_scaler=row["Scaler"] #Scaler for fitness 
            
            #get ground truth
            sel_eval_input["folder"]=sim_folder #update folder
            sel_eval_input["names_sim_repeat"]=names_opt #update names
            gt_value=sel_eval(**sel_eval_input) #same sims are evaluated for independent parameter sets

            #append ground truth
            row["f train"].append(gt_value) #add to list 

            #get rescaled ground truth/fitness
            scaler_gt=sel_scaler #()
            f_train_rs= scaler_gt.fit_transform(np.asarray(row["f train"]).reshape(-1, 1)).flatten()
            #print("updated f train",len(f_train_rs))
            #add rs ground truth
            #have to manually insert entire rescaled f train array, cant use row[""] syntax, have to use .at
            id_sel_set=df_eval_opt_param_set[df_eval_opt_param_set["Set of indep params"]==sel_set].index.values[0]
            df_eval_opt_param_set.at[id_sel_set,"f train rs"]=f_train_rs
            
            #add x_train and x_sim_values
            #row["GP params"]["x_train_rs"]=row["x train rs"]
            row["GP params"]["f_train_rs"]=f_train_rs 
            
    #save sim_doc_file
    if file_name_sim_params!=None:
        sim_doc_file_save_name=sim_folder+file_name_sim_params
        np.savetxt(sim_doc_file_save_name,sim_doc_file,delimiter=" ", fmt="%s")
    #for index, row in df_eval_opt_param_set.iterrows():
        #row["x train rs"].append("test2")
        #row["f train rs"].append("train test")
    return df_eval_opt_param_set, gpr
    



##########
#
##########






##########
#
##########    