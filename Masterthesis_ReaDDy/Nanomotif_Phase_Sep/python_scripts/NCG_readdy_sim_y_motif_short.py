#functions used in NCG_param_opt_y_motif_BoTorch notebook

#Updated for short Y-motif

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import readdy
import math
import scipy
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import time
import scipy.integrate as integrate
#print(readdy.__version__)
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import Counter
import random
from functools import partial
#import tensorflow.compat.v2 as tf
#tf.enable_v2_behavior()
#from tensorflow.keras import datasets, layers, models, Model, Input
import pandas as pd

#from sklearn import metrics
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.decomposition import PCA
#from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d

from multiprocessing import Process, Manager



##########
#Functions for setting up coords and evaluating angles in ReaDDy sims of nanomotifs
##########

def randomvector(n,norm):
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = [norm*x/r for x in components]
    return v

def random_spherical_vol(n,r):
    i=0
    coords_l=[]
    while i<n:
        coords=np.random.random(size=(1, 3)) * 2*r - r
        coords_x= coords[0][0]
        coords_y= coords[0][1]
        coords_z= coords[0][2]
        if np.sqrt(coords_x**2+coords_y**2+coords_z**2)<r:
            i=i+1
            coords_l.append(coords[0])
    return np.asarray(coords_l)

def angle_1(v1,v2):

    v1_n=v1/np.linalg.norm(v1)
    v2_n=v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_n,v2_n),-1,1))
    
def cent_of_mass(list_of_coords):
    #list_of_coords= [[x1,y1,z1], [x2,y2,z2],...]
    x, y, z = np.array(list_of_coords).T
    cent = [np.mean(x), np.mean(y), np.mean(z)]
    return np.asarray(cent)

#get angles between vectors aligned to the arms
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


#get angles between vectors defined by two particles for each arm each
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
#only works if the two fused arms have unique fused monnomers (e.g. x_link_fused_a_1 and x_link_fused_c_1)
def conf_angles_2_bound_x_motifs_1(pos,types,traj,arms_names):
    
    #arms_names=[x_arm_(1)_,x_link_fused_(1)_,x_link_fused_(2)_,x_arm_(1)]
    #arms names are not unique (i.e. x_arm_(1)_exists twice) 
    #get indices of x_link_fused_(1/2)_, wich have to be unique for this to work, to find the closer corresponding arm index
    
    angles_fused_arms=[]
  
    #get correct index for x_arm_(1/2)_
    
    possible_index_x_arm_1=[]
    possible_index_x_arm_2=[]
    for i in range(len(pos[0])): #check first time step for both x_arm_(1/2)_ positions
        #print(traj.species_name(types[0][i]))
        if traj.species_name(types[0][i])==arms_names[0]:
            #print("!!")
            possible_index_x_arm_1.append(int(i))
            
        if traj.species_name(types[0][i])==arms_names[3]:
            possible_index_x_arm_2.append(int(i))
        
        if traj.species_name(types[0][i])==arms_names[1]:
            index_x_link_fused_1=int(i)
            
        if traj.species_name(types[0][i])==arms_names[2]:
            index_x_link_fused_2=int(i)
    #print("index",possible_index_x_arm_1,possible_index_x_arm_2,index_x_link_fused_1,index_x_link_fused_2)
    distance_possible_index_x_arm_1=[]
    distance_possible_index_x_arm_2=[]
    for i in range(len(possible_index_x_arm_1)): #get distances from unique fused indeces to all non-uniqe arms
        distance_possible_index_x_arm_1.append( np.sqrt(np.sum( (pos[0][possible_index_x_arm_1[i]] - pos[0][index_x_link_fused_1])**2  ) ) )
        distance_possible_index_x_arm_2.append( np.sqrt(np.sum( (pos[0][possible_index_x_arm_2[i]] - pos[0][index_x_link_fused_2])**2  ) ) )
        
    #print("dist",distance_possible_index_x_arm_1,distance_possible_index_x_arm_2)                                           
    #get index of shorter distance
    index_shorter_dist_arm_1=np.argmin(distance_possible_index_x_arm_1)
    index_shorter_dist_arm_2=np.argmin(distance_possible_index_x_arm_2)
    #print("index_short dist",index_shorter_dist_arm_1,index_shorter_dist_arm_2)                                          
    index_x_arm_1=int(possible_index_x_arm_1[index_shorter_dist_arm_1])
    index_x_arm_2=int(possible_index_x_arm_2[index_shorter_dist_arm_2])                                
    #print("index",index_x_arm_1,index_x_arm_2)                                     
    for i in range(len(pos)):
        #get centre of mass of two x_link_fused monomers
        cent_of_mass_x_link_fused=cent_of_mass( list_of_coords=[pos[i][index_x_link_fused_1],pos[i][index_x_link_fused_2]] )
        #cent_of_mass_x_link_fused=pos[i][index_x_link_fused_1]
        vec_1=cent_of_mass_x_link_fused - pos[i][index_x_arm_1]
        vec_2=pos[i][index_x_arm_2] -cent_of_mass_x_link_fused
        angles_fused_arms.append(angle_1(vec_1,vec_2))
                                               
    return np.asarray(angles_fused_arms)
    








#############
#Templates for y motif
#############

#template for y-motif, starting from centre, iterating trough arms,
#not regarding number of monomers, only type
template_y_motif_1_type=["y_centre_1",
                         "y_base_arm_a_1","y_arm_a_1","y_link_a_1",
                         "y_base_arm_b_1","y_arm_b_1","y_link_b_1",
                         "y_base_arm_c_1","y_arm_c_1","y_link_c_1"]

#get template with correct number of monomers, particle types can be repeated by setting the corresponding number to 2, 3...
template_y_motif_1_nums=[1,
                         1,1,1,
                         1,1,1,
                         1,1,1]
template_y_motif_1=[]
for i in range(len(template_y_motif_1_type)):
    num_of_type=template_y_motif_1_nums[i]
    for j in range(num_of_type):
        template_y_motif_1.append(template_y_motif_1_type[i])
        
#template for y-motif, with HP config, every monomer once
template_y_motif_1_HP_type=["y_centre_1",
                            "y_base_arm_a_1","y_arm_a_1","y_link_hp_a_1",
                            "y_base_arm_b_1","y_arm_b_1","y_link_hp_b_1",
                            "y_base_arm_c_1","y_arm_c_1","y_link_hp_c_1"]

#get template with correct number of monomers, particle types can be repeated by setting the corresponding number to 2, 3...
template_y_motif_1_nums=[1,
                         1,1,1,
                         1,1,1,
                         1,1,1]
template_y_motif_1_HP=[]
for i in range(len(template_y_motif_1_HP_type)): #same for type 1,2
    num_of_type=template_y_motif_1_nums[i]
    for j in range(num_of_type): 
        template_y_motif_1_HP.append(template_y_motif_1_HP_type[i])

#monomer types for base, arm, link
list_base_types_y_motif=["y_base_arm_a_1","y_base_arm_b_1","y_base_arm_c_1"]
list_arm_types_y_motif=["y_arm_a_1","y_arm_b_1","y_arm_c_1"]
list_link_types_y_motif=["y_link_a_1","y_link_b_1","y_link_c_1"]
list_link_fused_types_y_motif=["y_link_fused_a_1","y_link_fused_b_1","y_link_fused_c_1"]
list_link_hp_types_y_motif_1=["y_link_hp_a_1","y_link_hp_b_1","y_link_hp_c_1"]

#template y motif coords
def y_motif_template_coords(n):
    angles=[np.radians(0),np.radians(120),np.radians(240)]
    y_motifs_tc=[]
    #y_f1=np.asarray([(n,0,0),(2*n,0,0),(3*n,0,0)])
    
    #positions of monomers in one arm n-2n-3n-... monomer size
    c1=(n,0,0)
    c2=(2*n,0,0)
    c3=(3*n,0,0)
    for i in range(len(angles)):
        #new_coords=
        c1n=np.matmul(Rot_xy(angles[i]),c1)
        c2n=np.matmul(Rot_xy(angles[i]),c2)
        c3n=np.matmul(Rot_xy(angles[i]),c3)
        
        y_motifs_tc.append(c1n)
        y_motifs_tc.append(c2n)
        y_motifs_tc.append(c3n)
    y_motifs_tc.insert(0,np.asarray([0,0,0]))
    return np.asarray(y_motifs_tc)










    
#############
#Templates for x motif
#############

#x motif 1
#template for x-motif, starting from centre, iterating trough arms,
#not regarding number of monomers, only type
template_x_motif_1_type=["x_centre_1","x_base_arm_a_1","x_arm_a_1","x_link_a_1","x_base_arm_b_1","x_arm_b_1","x_link_b_1","x_base_arm_c_1","x_arm_c_1","x_link_c_1","x_base_arm_d_1","x_arm_d_1","x_surf_link_d_1"]

#get template with correct number of monomers, particle types can be repeated by setting the corresponding number to 2, 3...
template_x_motif_1_nums=[1,1,1,1,1,1,1,1,1,1,1,1,1]
template_x_motif_1=[]
for i in range(len(template_x_motif_1_type)): #same for type 1,2
    num_of_type=template_x_motif_1_nums[i]
    for j in range(num_of_type): 
        template_x_motif_1.append(template_x_motif_1_type[i])

#template for x-motif, with HP config, every monomer once
template_x_motif_1_HP_type=["x_centre_1","x_base_arm_a_1","x_arm_a_1","x_link_hp_a_1","x_base_arm_b_1","x_arm_b_1","x_link_hp_b_1","x_base_arm_c_1","x_arm_c_1","x_link_hp_c_1","x_base_arm_d_1","x_arm_d_1","x_surf_link_hp_d_1"]

#get template with correct number of monomers, particle types can be repeated by setting the corresponding number to 2, 3...
template_x_motif_1_nums=[1,1,1,1,1,1,1,1,1,1,1,1,1]
template_x_motif_1_HP=[]
for i in range(len(template_x_motif_1_HP_type)): #same for type 1,2
    num_of_type=template_x_motif_1_nums[i]
    for j in range(num_of_type): 
        template_x_motif_1_HP.append(template_x_motif_1_HP_type[i])

        
#monomer types for base, arm, link ,hp      
list_base_types_x_motif=["x_base_arm_a_1","x_base_arm_b_1","x_base_arm_c_1","x_base_arm_d_1"] #same for type 1,2             
list_arm_types_x_motif= ["x_arm_a_1","x_arm_b_1","x_arm_c_1","x_arm_d_1"] #same for type 1,2               
list_link_types_x_motif_1=["x_link_a_1","x_link_b_1","x_link_c_1","x_surf_link_d_1"]     
list_link_fused_types_x_motif_1=["x_link_fused_a_1","x_link_fused_b_1","x_link_fused_c_1","x_surf_link_fused_d_1"]
list_link_hp_types_x_motif_1=["x_link_hp_a_1","x_link_hp_b_1","x_link_hp_c_1","x_surf_link_hp_d_1"]
       
    
#x motif 2
#template for x-motif 2, starting from centre, iterating trough arms,
template_x_motif_2_type=["x_centre_1","x_base_arm_a_1","x_arm_a_1","x_link_a_2","x_base_arm_b_1","x_arm_b_1","x_link_b_2","x_base_arm_c_1","x_arm_c_1","x_link_c_2","x_base_arm_d_1","x_arm_d_1","x_surf_link_d_2"]

#get template with correct number of monomers, particle types can be repeated by setting the corresponding number to 2, 3...
template_x_motif_2_nums=[1,1,1,1,1,1,1,1,1,1,1,1,1]
template_x_motif_2=[]
for i in range(len(template_x_motif_2_type)): #same for type 1,2
    num_of_type=template_x_motif_2_nums[i]
    for j in range(num_of_type):
        template_x_motif_2.append(template_x_motif_2_type[i])   


#template for x-motif 2, with HP config, every monomer once
template_x_motif_2_HP_type=["x_centre_1","x_base_arm_a_1","x_arm_a_1","x_link_hp_a_2","x_base_arm_b_1","x_arm_b_1","x_link_hp_b_2","x_base_arm_c_1","x_arm_c_1","x_link_hp_c_2","x_base_arm_d_1","x_arm_d_1","x_surf_link_hp_d_2"]

#get template with correct number of monomers, particle types can be repeated by setting the corresponding number to 2, 3...
template_x_motif_2_nums=[1,1,1,1,1,1,1,1,1,1,1,1,1]
template_x_motif_2_HP=[]
for i in range(len(template_x_motif_2_HP_type)): #same for type 1,2
    num_of_type=template_x_motif_2_nums[i]
    for j in range(num_of_type):
        template_x_motif_2_HP.append(template_x_motif_2_HP_type[i])

#monomer types for base, arm, link ,hp  
#base and arm types identical to x motif 1
list_link_types_x_motif_2=["x_link_a_2","x_link_b_2","x_link_c_2","x_surf_link_d_2"]
list_link_fused_types_x_motif_2=["x_link_fused_a_2","x_link_fused_b_2","x_link_fused_c_2","x_surf_link_fused_d_2"]
list_link_hp_types_x_motif_2= ["x_link_hp_a_2","x_link_hp_b_2","x_link_hp_c_2","x_surf_link_hp_d_2"]


#template x motif coords
def x_motif_template_coords(n):
    angles=[np.radians(0),np.radians(90),np.radians(180),np.radians(270)]
    y_motifs_tc=[]
    #y_f1=np.asarray([(n,0,0),(2*n,0,0),(3*n,0,0)])
    
    #positions of monomers in one arm n-2n-3n-... monomer size
    c1=(n,0,0)
    c2=(2*n,0,0)
    c3=(3*n,0,0)
    for i in range(len(angles)):
        #new_coords=
        c1n=np.matmul(Rot_xy(angles[i]),c1)
        c2n=np.matmul(Rot_xy(angles[i]),c2)
        c3n=np.matmul(Rot_xy(angles[i]),c3)
        
        y_motifs_tc.append(c1n)
        y_motifs_tc.append(c2n)
        y_motifs_tc.append(c3n)
    y_motifs_tc.insert(0,np.asarray([0,0,0]))
    return np.asarray(y_motifs_tc)
        












#############
#Templates for surface
############

#surface layout option 1       
def fct_template_surface_1(num_rep_unit=4,num_a=3,num_b=4,num_c=3):
    
    template_surface_1=[]

    for k in range(num_rep_unit):
        for i in range(num_a):
            template_surface_1.append("x_surf_fill_1")

        for i in range(num_b):
            template_surface_1.append("x_surf_int_1")

        for i in range(num_c):
            template_surface_1.append("x_surf_fill_1")
    return template_surface_1

template_surface_1=fct_template_surface_1()

list_surface_type_1=["x_surf_fill_1","x_surf_int_1","x_surf_bound_int_1"]
list_surface_type_1_ss= ["x_surf_int_1"]
list_surface_type_1_ds=["x_surf_fill_1","x_surf_bound_int_1"]


def surf_template_coords(n):
        
    surf_ini=[(0,0,0)]
    
    pos_list_r=np.zeros((len(template_surface_1),3))

    pos_list_r[0]=surf_ini[0]
    norm_v=n #spacing between monomers
    m=0
    while m < len(template_surface_1)-1:
        
        pos_e=randomvector(3,norm_v)+pos_list_r[m] #pos at [m+1]
        #pos_e=[(2.72,0,0)]+pos_list_r[n-1]
        
        #self avoidance, check if other positions closer than norm_v to new position
        pos_val=True
        for l in range(m-1):
            if np.sqrt( (pos_e[0] - pos_list_r[l][0])**2 + (pos_e[1] - pos_list_r[l][1])**2 + (pos_e[2] - pos_list_r[l][2])**2 )<norm_v:
                pos_val=False
                break
                
        if pos_val==True:
            pos_list_r[m+1]=np.asarray(pos_e)
            m=m+1

    return np.asarray(pos_list_r)

#template surface
#coordinates within box volume
def surf_template_coords_2(n,box_vol):
    
    surf_ini=[(0,0,0)]
    
    pos_list_r=np.zeros((len(template_surface_1),3))

    pos_list_r[0]=surf_ini[0]
    norm_v=n #spacing between monomers
    m=0
    while m < len(template_surface_1)-1:
        
        pos_e=randomvector(3,norm_v)+pos_list_r[m] #pos at [m+1]
        #pos_e=[(2.72,0,0)]+pos_list_r[n-1]
        
        #self avoidance, check if other positions closer than norm_v to new position
        pos_val=True
        for l in range(m-1):
            if np.sqrt( (pos_e[0] - pos_list_r[l][0])**2 + (pos_e[1] - pos_list_r[l][1])**2 + (pos_e[2] - pos_list_r[l][2])**2 )<norm_v:
                pos_val=False
                break
        #check if new coordinates within box volume       
        box_val=False
        if (np.abs(pos_e[0]) < box_vol[0]) and (np.abs(pos_e[1]) < box_vol[1]) and (np.abs(pos_e[2]) < box_vol[2]):
            box_val=True
                
        if pos_val==True and box_val==True:
            pos_list_r[m+1]=np.asarray(pos_e)
            m=m+1

    return np.asarray(pos_list_r)












###############
#list of all types of particles
###############
all_types_d=[*template_y_motif_1_type,
             *list_link_fused_types_y_motif,
             *list_link_hp_types_y_motif_1,
             *template_x_motif_1_type,
             *list_link_fused_types_x_motif_1,
             *template_x_motif_2_type,
             *list_link_fused_types_x_motif_2,
             *list_link_hp_types_x_motif_1,
             *list_link_hp_types_x_motif_2,
             *list_surface_type_1]

all_types=pd.Series(all_types_d).unique().tolist()

###############
#list of all ss-DNA particles
###############
list_ss_DNA_particles_a=[]
#append y motif particles
list_ss_DNA_particles_a.append([*list_link_types_y_motif])

#append x motif 1 particles
list_ss_DNA_particles_a.append([*list_link_types_x_motif_1]  )

#append x motif 2 particles
list_ss_DNA_particles_a.append([*list_link_types_x_motif_2]  )

#append surface particles

list_ss_DNA_particles_a.append([*list_surface_type_1_ss])

#flat list
list_ss_DNA_particles_1=[]
for i in range(len(list_ss_DNA_particles_a)):
    for j in range(len(list_ss_DNA_particles_a[i])):
        list_ss_DNA_particles_1.append(list_ss_DNA_particles_a[i][j])       
   
   
#rotation matrix, rotate in x-y plane by angle
def Rot_xy(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s,0), (s, c,0),(0,0,1)))
    











#############
#Physical params of nanomotifs
#############

#definition of Debye length    
def Debye_length(L,T):
    #L = ionic strength, for monovalent ion concentration times charge
    kbt=T*1.380649*10**-23
    T_C=T-273.15 # temp in °C for e_r calculation of water
    eps_0=8.8541878128 * 10**-12 #F/m
    eps_r=87.740 - 0.40008*T_C + 9.398*10**-4 * T_C**2 -1.41 *10**-6 * T_C**3 #https://nvlpubs.nist.gov/nistpubs/jres/56/jresv56n1p1_a1b.pdf
        
    D_l=np.sqrt(eps_0*eps_r*kbt/(2*L))
    return D_l

#repulsion due to DB potential
def Debye_Hueckel_pot_rep_strength(q1,q2,T):
    Q1b=q1*1.602176634 * 10**-19
    Q2b=q2*1.602176634 * 10**-19
    T_C=T-273.15
    eps_0=8.8541878128 * 10**-12 #F/m
    eps_r=87.740 - 0.40008*T_C + 9.398*10**-4 * T_C**2 -1.41 *10**-6 * T_C**3 #https://nvlpubs.nist.gov/nistpubs/jres/56/jresv56n1p1_a1b.pdf
    
    C=Q1b*Q2b /(4*np.pi*eps_0*eps_r) #[C]=Joule*metre
    return C
    
 
#Diffusion constant
# viscositiy: https://srd.nist.gov/jpcrdreprint/1.555581.pdf
# scaling diffusion constant with T: https://en.wikipedia.org/wiki/Einstein_relation_(kinetic_theory)
# Diffusion constant DNA oligos at 23°C: https://www.sciencedirect.com/science/article/pii/S0021925818310871

def Diff_at23C(num_bp):
    return 0.49 * 1/num_bp**0.72 #nm^2/ns



def visc_T(T_in):
    T=T_in-273.15
    mu_0=1002.0 *10**-6
    mu_T=mu_0 * 10**( (20-T)/(T+96) * (1.2387 - 1.303*10**-3 * (20-T) + 3.06*10**-6 * (20-T)**2 + 2.55*10**-8 * (20-T)**3 ) )
    
    return mu_T  #Pa*s
    


def Diff_ES(T,r,eta):
    D=T*1.380649*10**-23 /(6*np.pi*eta*r)
    return D   #m^2/s   
            
  


  
#############
#ReaDDy reactions x/y/surf motifs
#############

#apply reaction at rate k
def param_rate_fission_selection(topology,k):
    return k

#fision reaction for y motifs, type 1 and type 2 x motifs, no cross reaction between them
#exp_rate_ub_x/y -> fission of x_1/2 or y motif
#exp_rate_ub_x_surf_1/2 -> fission of x 1/2 and surface
def param_fission_5(topology,exp_rate_ub_y_1,exp_rate_ub_x_1,exp_rate_ub_x_surf_1,exp_rate_ub_x_2,exp_rate_ub_x_surf_2,
                    
                    list_link_fused_types_y_motif,list_link_fused_types_x_motif_1,list_link_fused_types_x_motif_2):
                    
    recipe = readdy.StructuralReactionRecipe(topology)
    
    #vertices = topology.get_graph().get_vertices()
    
    edges = topology.get_graph().get_edges()
   
    for e in edges:
        v1, v2 = e[0], e[1]
        #get names of particles
        particle_name1=topology.particle_type_of_vertex(v1)
        particle_name2=topology.particle_type_of_vertex(v2)
         
        ##### y motif #####
        #if particles are of linked type (irregardless if a,b,c) make reaction
        #use exp_rate_ub_y_1 for y-link fission
        if (particle_name1 in list_link_fused_types_y_motif) and (particle_name2 in list_link_fused_types_y_motif):
            if np.random.rand()<exp_rate_ub_y_1:
                #change y_link_fused_x_1 to y_link_x_1:
                #particles at vertice 1
                if particle_name1=="y_link_fused_a_1":
                    recipe.change_particle_type(v1, "y_link_a_1")
                    
                if particle_name1=="y_link_fused_b_1":
                    recipe.change_particle_type(v1, "y_link_b_1")
                    
                if particle_name1=="y_link_fused_c_1":
                    recipe.change_particle_type(v1, "y_link_c_1")
                
                #particles at vertices 2
                if particle_name2=="y_link_fused_a_1":
                    recipe.change_particle_type(v2, "y_link_a_1")
                    
                if particle_name2=="y_link_fused_b_1":
                    recipe.change_particle_type(v2, "y_link_b_1")
                    
                if particle_name2=="y_link_fused_c_1":
                    recipe.change_particle_type(v2, "y_link_c_1")
                    
                recipe.remove_edge(v1,v2)
                
        ###### x-motif 1 #####
        #use exp_rate_ub_x_1 for x-link-1 fission of self interaction arms
        if (particle_name1 in list_link_fused_types_x_motif_1) and (particle_name2 in list_link_fused_types_x_motif_1):
            if np.random.rand()<exp_rate_ub_x_1:
                #change y_link_fused_x_1 to y_link_x_1:
                #particles at vertice 1
                if particle_name1=="x_link_fused_a_1":
                    recipe.change_particle_type(v1, "x_link_a_1")
                    
                if particle_name1=="x_link_fused_b_1":
                    recipe.change_particle_type(v1, "x_link_b_1")
                    
                if particle_name1=="x_link_fused_c_1":
                    recipe.change_particle_type(v1, "x_link_c_1")
                
                #particles at vertices 2
                if particle_name2=="x_link_fused_a_1":
                    recipe.change_particle_type(v2, "x_link_a_1")
                    
                if particle_name2=="x_link_fused_b_1":
                    recipe.change_particle_type(v2, "x_link_b_1")
                    
                if particle_name2=="x_link_fused_c_1":
                    recipe.change_particle_type(v2, "x_link_c_1")
                    
                recipe.remove_edge(v1,v2)
                
        #use exp_rate_ub_x_surf_1 for x-link-1 fission of surface
        if (particle_name1 == "x_surf_bound_int_1" and particle_name2 == "x_surf_link_fused_d_1") or (particle_name2 == "x_surf_bound_int_1" and particle_name1 == "x_surf_link_fused_d_1") :
            if np.random.rand()<exp_rate_ub_x_surf_1:
                #particle at vertice 1 is surface
                if particle_name1=="x_surf_bound_int_1":
                    recipe.change_particle_type(v1, "x_surf_int_1")
                    #other particle must be x_surf_link_fused_d_1 in this case
                    recipe.change_particle_type(v2, "x_surf_link_d_1")
                    
                #particle at vertice 1 is x-link
                if particle_name1=="x_surf_link_fused_d_1":
                    recipe.change_particle_type(v1, "x_surf_link_d_1")
                    #other particle must be x_surf_link_fused_d_1 in this case
                    recipe.change_particle_type(v2, "x_surf_int_1")
                    
                recipe.remove_edge(v1,v2)

        ###### x-motif 2 #####
        #use exp_rate_ub_x_2 for x-link-2 fission of self interaction arms
        if (particle_name1 in list_link_fused_types_x_motif_2) and (particle_name2 in list_link_fused_types_x_motif_2):
            if np.random.rand()<exp_rate_ub_x_2:
                #change y_link_fused_x_2 to y_link_x_2:
                #particles at vertice 1
                if particle_name1=="x_link_fused_a_2":
                    recipe.change_particle_type(v1, "x_link_a_2")
                    
                if particle_name1=="x_link_fused_b_2":
                    recipe.change_particle_type(v1, "x_link_b_2")
                    
                if particle_name1=="x_link_fused_c_2":
                    recipe.change_particle_type(v1, "x_link_c_2")
                
                #particles at vertices 2
                if particle_name2=="x_link_fused_a_2":
                    recipe.change_particle_type(v2, "x_link_a_2")
                    
                if particle_name2=="x_link_fused_b_2":
                    recipe.change_particle_type(v2, "x_link_b_2")
                    
                if particle_name2=="x_link_fused_c_2":
                    recipe.change_particle_type(v2, "x_link_c_2")
                    
                recipe.remove_edge(v1,v2)
                
        #use exp_rate_ub_x_surf_2 for x-link-2 fission of surface
        if (particle_name1 == "x_surf_bound_int_1" and particle_name2 == "x_surf_link_fused_d_2") or (particle_name2 == "x_surf_bound_int_1" and particle_name1 == "x_surf_link_fused_d_2") :
            if np.random.rand()<exp_rate_ub_x_surf_2:
                #particle at vertice 1 is surface
                if particle_name1=="x_surf_bound_int_1":
                    recipe.change_particle_type(v1, "x_surf_int_1")
                    #other particle must be x_surf_link_fused_d_2 in this case
                    recipe.change_particle_type(v2, "x_surf_link_d_2")
                    
                #particle at vertice 1 is x-link
                if particle_name1=="x_surf_link_fused_d_2":
                    recipe.change_particle_type(v1, "x_surf_link_d_2")
                    #other particle must be x_surf_link_fused_d_2 in this case
                    recipe.change_particle_type(v2, "x_surf_int_1")
                
                recipe.remove_edge(v1,v2)  
    return recipe

##
#generate fusion reactions between all particles
def gen_fusion_reactions_1(fusion_input_particles,fusion_output_particles,rate,distance,system):
    #iterate over all particles for particle 1
    for i in range(len(fusion_input_particles)):
        fusion_input_particle_1=fusion_input_particles[i]
        fusion_output_particle_1=fusion_output_particles[i]
        
        #iterate over all particles for particle 2
        for j in range(len(fusion_input_particles)):
            fusion_input_particle_2=fusion_input_particles[j]
            fusion_output_particle_2=fusion_output_particles[j]  
            
            fusion_name="Fusion_"+str(fusion_input_particle_1)+"_"+str(fusion_input_particle_2)
            
            fusion_reg = f"{fusion_name}: Particle({fusion_input_particle_1}) + Particle({fusion_input_particle_2}) -> Particle({fusion_output_particle_1}--{fusion_output_particle_2}) [self=true]"
        
            #print(fusion_reg)
    
            system.topologies.add_spatial_reaction(fusion_reg, rate=rate, radius=distance) 
##





###########
#ReaDDy potentials for surface, x and y motifs
###########

#add ReaDDy potentials for y-motif (no Volume exclusion)
def add_y_motif_potentials(system,
                           list_base_types_y_motif,
                           list_arm_types_y_motif,
                           list_link_types_y_motif,
                           list_link_fused_types_y_motif,
                           list_link_hp_types_y_motif_1,
                       
                           k_bond_ds,k_bond_ss,eq_dist_bb,eq_dist_bb_s,eq_dist_fused_bb,
                       
                           k_angle_ds_base,
                           k_angle_ds_arm,
                           k_angle_ss_link,
                           k_angle_ds_hp,
                           k_angle_ds_fused_link,
                       
                           theta_base_y,theta_0): 
    
    for i in range(len(list_base_types_y_motif)):
        #base type
        base_type_1=list_base_types_y_motif[i]
        #arm type
        arm_type_1=list_arm_types_y_motif[i]
        #link type
        link_type_1=list_link_types_y_motif[i]
        #fused link type
        link_fused_type_1=list_link_fused_types_y_motif[i]
        
        #bonds in motif
        system.topologies.configure_harmonic_bond(arm_type_1, base_type_1, force_constant=k_bond_ds, length=eq_dist_bb)
        system.topologies.configure_harmonic_bond("y_centre_1", base_type_1, force_constant=k_bond_ds, length=eq_dist_bb)
        
        #bonds involving sticky end
        system.topologies.configure_harmonic_bond(arm_type_1, link_type_1, force_constant=k_bond_ss, length=eq_dist_bb_s)
        system.topologies.configure_harmonic_bond(arm_type_1, link_fused_type_1, force_constant=k_bond_ds, length=eq_dist_bb_s)
        
    #Hairpin bonds
    for i in range(len(list_link_hp_types_y_motif_1)): # add bonds to arms if sticky ends on arms a,b,c form hairpins in y motif
        arm_type_1=list_arm_types_y_motif[i]
        link_hp_type_1=list_link_hp_types_y_motif_1[i]
        system.topologies.configure_harmonic_bond(arm_type_1, link_hp_type_1, force_constant=k_bond_ds, length=eq_dist_bb_s)
        
    #Bond between fused linkers
    for i in range(len(list_link_fused_types_y_motif)):
        link_fused_type_1=list_link_fused_types_y_motif[i]
        for j in range(i+1):
            #bond between fused linkers
            link_fused_type_2=list_link_fused_types_y_motif[j]
            system.topologies.configure_harmonic_bond(link_fused_type_1, link_fused_type_2, force_constant=k_bond_ds, length=eq_dist_fused_bb)


    #####################################

    #add angle potentials y-motif-1
    #add angles around centres

    for i in range(len(list_base_types_y_motif)):
        k=(i+1)%len(list_base_types_y_motif)
        #base type
        base_type_1=list_base_types_y_motif[i]
        base_type_2=list_base_types_y_motif[k]

        #arm type
        arm_type_1=list_arm_types_y_motif[i]

        #link type
        link_type_1=list_link_types_y_motif[i]
        #fused link type
        link_fused_type_1=list_link_fused_types_y_motif[i]    

        #k for base-centre-base angle
        system.topologies.configure_harmonic_angle(base_type_1, "y_centre_1", base_type_2, force_constant=k_angle_ds_base, equilibrium_angle=theta_base_y)
        system.topologies.configure_harmonic_angle("y_centre_1", base_type_1, arm_type_1, force_constant=k_angle_ds_arm, equilibrium_angle=theta_0)

        #k for base-arm-linker angle
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_type_1, force_constant=k_angle_ss_link, equilibrium_angle=theta_0)
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_fused_type_1, force_constant=k_angle_ss_link, equilibrium_angle=theta_0)

        #arm-linker_fused-linker_fused angle
        for j in range(len(list_link_fused_types_y_motif)):
            link_fused_type_2=list_link_fused_types_y_motif[j]
            system.topologies.configure_harmonic_angle(arm_type_1,link_fused_type_1, link_fused_type_2, force_constant=k_angle_ds_fused_link, equilibrium_angle=theta_0)
    
    #hairpin angles    
    for i in range(len(list_link_hp_types_y_motif_1)): # add angles involving hairpin
        base_type_1=list_base_types_y_motif[i]
        arm_type_1=list_arm_types_y_motif[i]
        link_hp_type_1=list_link_hp_types_y_motif_1[i]
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_hp_type_1, force_constant=k_angle_ds_hp, equilibrium_angle=theta_0)
     
     
     
# Add ReaDDy potentials for x motif (includes x motif 1 and 2 with same parameters) (no Volume exclusion)
def add_x_motif_potentials(system,list_base_types_x_motif,list_arm_types_x_motif,list_link_types_x_motif_1,list_link_types_x_motif_2,
                           
                           list_link_fused_types_x_motif_1,list_link_fused_types_x_motif_2,
                           
                           list_link_hp_types_x_motif_1,list_link_hp_types_x_motif_2,
                           
                           k_bond_ds,k_bond_ss,eq_dist_bb,eq_dist_bb_s,eq_dist_fused_bb,
                           
                           k_angle_ds_base,k_angle_ds_arm,k_angle_ss_link,k_angle_ds_hp,k_angle_ds_fused_link,k_angle_ds_oppa,
                           
                           theta_base_x,theta_0,theta_oppa_x):
                           
   #add bond potentials x-motif-1 and x-motif-2
    for i in range(len(list_base_types_x_motif)):
        #base type
        base_type_1=list_base_types_x_motif[i]
        #arm type
        arm_type_1=list_arm_types_x_motif[i]
        #link type x-motif-1
        link_type_1=list_link_types_x_motif_1[i]
        #fused link type x-motif-1
        link_fused_type_1=list_link_fused_types_x_motif_1[i]
        ###
        #link type x-motif-2
        link_type_2=list_link_types_x_motif_2[i]
        #fused link type x-motif-2
        link_fused_type_2=list_link_fused_types_x_motif_2[i] 

        #bonds in motif, x-motif-1 and x-motif-2
        system.topologies.configure_harmonic_bond(arm_type_1, base_type_1, force_constant=k_bond_ds, length=eq_dist_bb)
        system.topologies.configure_harmonic_bond("x_centre_1", base_type_1, force_constant=k_bond_ds, length=eq_dist_bb)
        #print("har. bonds x-motifs 1:",arm_type_1, base_type_1)
        #print("har. bonds x-motifs 2:","x_centre_1", base_type_1)
        #bonds involving sticky end
        #x-motif-1
        system.topologies.configure_harmonic_bond(arm_type_1, link_type_1, force_constant=k_bond_ss, length=eq_dist_bb_s)
        system.topologies.configure_harmonic_bond(arm_type_1, link_fused_type_1, force_constant=k_bond_ds, length=eq_dist_bb_s)
        #print("har. bonds x-motifs 3:",arm_type_1, link_type_1)
        #print("har. bonds x-motifs 4:",arm_type_1, link_fused_type_1)
        #x-motif-2
        system.topologies.configure_harmonic_bond(arm_type_1, link_type_2, force_constant=k_bond_ss, length=eq_dist_bb_s)
        system.topologies.configure_harmonic_bond(arm_type_1, link_fused_type_2, force_constant=k_bond_ds, length=eq_dist_bb_s)
        #print("har. bonds x-motifs 5:",arm_type_1, link_type_2)
        #print("har. bonds x-motifs 6:",arm_type_1, link_fused_type_2) 

    #hairpin bonds    
    for i in range(len(list_link_hp_types_x_motif_1)): # add bonds to arms if sticky ends on arms a,b,c,d form hairpins in x motifs 1 and 2
        arm_type_1=list_arm_types_x_motif[i]
        link_hp_type_1=list_link_hp_types_x_motif_1[i]
        link_hp_type_2=list_link_hp_types_x_motif_2[i]
        system.topologies.configure_harmonic_bond(arm_type_1, link_hp_type_1, force_constant=k_bond_ds, length=eq_dist_bb_s)
        system.topologies.configure_harmonic_bond(arm_type_1, link_hp_type_2, force_constant=k_bond_ds, length=eq_dist_bb_s)
        #print("har. bonds x-motifs 6:",arm_type_1, link_hp_type_1)    
        #print("har. bonds x-motifs 7:",arm_type_1, link_hp_type_2)     


    #fused types x-motif-1   
    for i in range(len(list_link_fused_types_x_motif_1[:3])): #only sticky ends a,b,c, not surface link d
        link_fused_type_1=list_link_fused_types_x_motif_1[i]
        for j in range(i+1):
            link_fused_type_2=list_link_fused_types_x_motif_1[j]
            system.topologies.configure_harmonic_bond(link_fused_type_1, link_fused_type_2, force_constant=k_bond_ds, length=eq_dist_fused_bb)           
            #print("har. bonds x-motifs 8:",link_fused_type_1, link_fused_type_2)
    #fused types x-motif-2  
    for i in range(len(list_link_fused_types_x_motif_2[:3])):
        link_fused_type_1=list_link_fused_types_x_motif_2[i] #only sticky ends a,b,c, not surface link d
        for j in range(i+1):
            link_fused_type_2=list_link_fused_types_x_motif_2[j]
            system.topologies.configure_harmonic_bond(link_fused_type_1, link_fused_type_2, force_constant=k_bond_ds, length=eq_dist_fused_bb)           
            #print("har. bonds x-motifs 9:",link_fused_type_1, link_fused_type_2)

    #add angle potentials x-motif-1 and x-motif-2
    #add angles around centres

    for i in range(len(list_base_types_x_motif)):
        k=(i+1)%len(list_base_types_x_motif)
        #base type
        base_type_1=list_base_types_x_motif[i]
        base_type_2=list_base_types_x_motif[k]

        #arm type
        arm_type_1=list_arm_types_x_motif[i]

        #link type x-motif-1
        link_type_1=list_link_types_x_motif_1[i]
        #fused link type x-motif-1
        link_fused_type_1=list_link_fused_types_x_motif_1[i] 
        ##
        #link type x-motif-2
        link_type_2=list_link_types_x_motif_2[i]
        #fused link type x-motif-2
        link_fused_type_2=list_link_fused_types_x_motif_2[i]  

        #base-centre-base angle
        system.topologies.configure_harmonic_angle(base_type_1, "x_centre_1", base_type_2, force_constant=k_angle_ds_base, equilibrium_angle=theta_base_x)
        #print("angle bonds x-motifs 1:",base_type_1, "x_centre_1", base_type_2)

        system.topologies.configure_harmonic_angle("x_centre_1", base_type_1, arm_type_1, force_constant=k_angle_ds_arm, equilibrium_angle=theta_0)
        #print("angle bonds x-motifs 2:","x_centre_1", base_type_1, arm_type_1)

        #base-arm-linker angle x-motif-1
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_type_1, force_constant=k_angle_ss_link, equilibrium_angle=theta_0)
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_fused_type_1, force_constant=k_angle_ss_link, equilibrium_angle=theta_0)
        #print("angle bonds x-motifs 3:",base_type_1, arm_type_1, link_type_1)
        #print("angle bonds x-motifs 4:",base_type_1, arm_type_1, link_fused_type_1)

        #base-arm-linker angle x-motif-2
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_type_2, force_constant=k_angle_ss_link, equilibrium_angle=theta_0)
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_fused_type_2, force_constant=k_angle_ss_link, equilibrium_angle=theta_0)
        #print("angle bonds x-motifs 5:",base_type_1, arm_type_1, link_type_2)
        #print("angle bonds x-motifs 6:",base_type_1, arm_type_1, link_fused_type_2)

    #hairpin angles    
    for i in range(len(list_link_hp_types_x_motif_1)): # add bonds to arms if sticky ends on arms a,b,c,d form hairpins in x motifs 1 and 2
        base_type_1=list_base_types_x_motif[i]
        arm_type_1=list_arm_types_x_motif[i]
        link_hp_type_1=list_link_hp_types_x_motif_1[i]
        link_hp_type_2=list_link_hp_types_x_motif_2[i]
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_hp_type_1, force_constant=k_angle_ds_hp, equilibrium_angle=theta_0)
        system.topologies.configure_harmonic_angle(base_type_1, arm_type_1, link_hp_type_2, force_constant=k_angle_ds_hp, equilibrium_angle=theta_0)
        #print("angle bonds x-motifs 7:",base_type_1, arm_type_1, link_hp_type_1)
        #print("angle bonds x-motifs 8:",base_type_1, arm_type_1, link_hp_type_2)

    #arm-linker_fused-linker_fused angle
    #x motif 1
    for i in range(len(list_base_types_x_motif[:3])): #only select the first three arms wich have self interacting sticky ends
        #arm type
        arm_type_1=list_arm_types_x_motif[i]
        #fused link type x-motif-1
        link_fused_type_1=list_link_fused_types_x_motif_1[i]
        for j in range(len(list_link_fused_types_x_motif_1[:3])): #all three x_linked_fused variante (a,b,c)
            link_fused_type_2=list_link_fused_types_x_motif_1[j]
            #print("angle bonds x-motifs 9:",arm_type_1,link_fused_type_1,link_fused_type_2)
            system.topologies.configure_harmonic_angle(arm_type_1, link_fused_type_1, link_fused_type_2, force_constant=k_angle_ds_fused_link, equilibrium_angle=theta_0)

    #x motif 2
    for i in range(len(list_base_types_x_motif[:3])): #only select the first three arms wich have self interacting sticky ends
        #arm type
        arm_type_1=list_arm_types_x_motif[i]
        #fused link type x-motif-1
        link_fused_type_1=list_link_fused_types_x_motif_2[i]
        for j in range(len(list_link_fused_types_x_motif_2[:3])): #all three x_linked_fused variante (a,b,c)
            link_fused_type_2=list_link_fused_types_x_motif_2[j]
            #print("angle bonds x-motifs 10:",arm_type_1,link_fused_type_1,link_fused_type_2)  
            system.topologies.configure_harmonic_angle(arm_type_1, link_fused_type_1, link_fused_type_2, force_constant=k_angle_ds_fused_link, equilibrium_angle=theta_0)
    
    #add opposing arm angle for both motif 1 and 2
    #arm 1-3:
    system.topologies.configure_harmonic_angle("x_base_arm_a_1", "x_centre_1", "x_base_arm_c_1", force_constant=k_angle_ds_oppa, equilibrium_angle=theta_oppa_x)
    #arm 2-4:
    system.topologies.configure_harmonic_angle("x_base_arm_b_1", "x_centre_1", "x_base_arm_d_1", force_constant=k_angle_ds_oppa, equilibrium_angle=theta_oppa_x)

    
#Add ReaDDy potentials for DNA surface (no volume exclusion, no surface-nanomotif potentials)
def add_surface_potentials(system,list_surface_type_1,list_surface_type_1_ss,list_surface_type_1_ds,
                          
                           k_bond_ds,k_bond_ss,eq_dist_bb,
                           
                           k_angle_ss_surf,k_angle_ds_surf,
                           
                           theta_0):
    
    #SURFACE
    #add bond potentials
    for i in range(len(list_surface_type_1)):
        for j in range(i+1):
            #both ds DNA
            if (list_surface_type_1[i] in list_surface_type_1_ds) and (list_surface_type_1[j] in list_surface_type_1_ds):
                system.topologies.configure_harmonic_bond(list_surface_type_1[i], list_surface_type_1[j], force_constant=k_bond_ds, length=eq_dist_bb)           

            #ss DNA (treat ss-ds and ss-ss same here)
            else:
                system.topologies.configure_harmonic_bond(list_surface_type_1[i], list_surface_type_1[j], force_constant=k_bond_ss, length=eq_dist_bb)           

    #add angles in surface
    for i in range(len(list_surface_type_1)):
        for j in range(len(list_surface_type_1)): #catch all permutations!!
            for k in range(len(list_surface_type_1)): #catch all permutations!!
                t1=list_surface_type_1[i]
                t2=list_surface_type_1[j]
                t3=list_surface_type_1[k]
                #treat as ss-DNA when 2 or 3 monomers are ss-DNA
                if (t1 in list_surface_type_1_ss and t2 in list_surface_type_1_ss) or (t1 in list_surface_type_1_ss and t3 in list_surface_type_1_ss) or (t2 in list_surface_type_1_ss and t3 in list_surface_type_1_ss):
                    system.topologies.configure_harmonic_angle(t1,t2,t3,force_constant=k_angle_ss_surf, equilibrium_angle=theta_0)
                if (t1 in list_surface_type_1_ss and t2 in list_surface_type_1_ss and t3 in list_surface_type_1_ss):
                    system.topologies.configure_harmonic_angle(t1,t2,t3,force_constant=k_angle_ss_surf, equilibrium_angle=theta_0)
                #1 or 0 ss-DNA monomers
                else:
                    system.topologies.configure_harmonic_angle(t1,t2,t3,force_constant=k_angle_ds_surf, equilibrium_angle=theta_0)

                
#Add ReaDDy potentials for particle-particle interaction e.g. Debye-Hückel repulsion                
def add_interaction_potentials(system,all_types,list_ss_DNA_particles_1,
                               
                               volume_excl_type,
                              
                               k_ex_ss_ds,k_ex_ss_ss,k_ex_ds_ds,
                               
                               DH_strength_ss_ds,DH_strength_ss_ss,DH_strength_ds_ds,DH_len,DH_cutoff): 
    
    #REPULSION
    #list of ss-DNA particles
    list_ss_DNA_particles=list_ss_DNA_particles_1
    for i in range(len(all_types)):
        for j in range(i+1):

            #if one particle is ss-DNA
            if  ((all_types[i] in list_ss_DNA_particles) and (all_types[j] not in list_ss_DNA_particles)):
                #print("ss-ds",all_types[i],all_types[j])
                #har rep
                k_hr_c=k_ex_ss_ds
                #DH pot
                DH_strength=DH_strength_ss_ds

            if  ((all_types[j] in list_ss_DNA_particles) and (all_types[i] not in list_ss_DNA_particles)):
                #print("ss-ds",all_types[i],all_types[j])
                #har rep
                k_hr_c=k_ex_ss_ds
                #DH pot
                DH_strength=DH_strength_ss_ds

            #if both particles in ss-DNA
            if  ((all_types[j] in list_ss_DNA_particles) and (all_types[i] in list_ss_DNA_particles)):
                #print("ss-ss",all_types[i],all_types[j])
                #har rep
                k_hr_c=k_ex_ss_ss
                #DH pot
                DH_strength=DH_strength_ss_ss

            #both particles ds-DNA    
            if  ((all_types[j] not in list_ss_DNA_particles) and (all_types[i] not in list_ss_DNA_particles)): 
                #print("ds-ds",all_types[i],all_types[j])
                #har rep
                k_hr_c=k_ex_ds_ds
                #DH pot
                DH_strength=DH_strength_ds_ds
            if volume_excl_type=="har_rep":
                system.potentials.add_harmonic_repulsion(all_types[i], all_types[j], force_constant=k_hr_c, interaction_distance=dist_ex)

            if volume_excl_type=="DH":
                       system.potentials.add_screened_electrostatics(
                        all_types[i], all_types[j], electrostatic_strength=DH_strength, inverse_screening_depth=1/DH_len,
                        repulsion_strength=0, repulsion_distance=1, exponent=0, cutoff=DH_cutoff)
  
 
#Add y/x_1/x_2/surface to sim with start pos and option for rotation    
def add_particles_to_sim(simulation,num_y_motif_1,y_motif_1_ini,y_motif_1_rot,template_y_motif_1,template_y_motif_1_HP,y_motif_template_coords,
                        
                         num_x_motif_1,x_motif_1_ini,x_motif_1_rot,template_x_motif_1,template_x_motif_1_HP, 
                         
                         num_x_motif_2,x_motif_2_ini,x_motif_2_rot,template_x_motif_2,template_x_motif_2_HP, x_motif_template_coords,
                         
                         use_y_1_HP_template,use_x_1_HP_template,use_x_2_HP_template,
                         
                         num_surface_1,surf_1_ini,template_surface_1,surf_template_coords):
    #add y-motif-1
    #generate coord template
    y_motif_1_tc=y_motif_template_coords(2.72)
    #different offset for each  particle
    for k in range(num_y_motif_1):
        #y_motif_1_ini=random_spherical_vol(1,r_sp/2)
        #y_motif_1_ini=random_spherical_vol(1,30)
        #y_motif_1_ini= np.random.uniform(size=(1,3)) * np.asarray([120,120,120])/5 - np.asarray([120,120,120])/5 * 0.5
        #y_motif_1_ini=np.asarray([[0,0,0]])
        #y_motif_1_ini=np.random.uniform(size=(1,3)) * system.box_size.magnitude - system.box_size.magnitude * 0.5

        pos_list_y_motif_1=[]
        if y_motif_1_rot==True:
            Rot_1=Rot.random()
        for m in range(len(y_motif_1_tc)):
            #rotation
            if y_motif_1_rot==True:
                pos_list_y_motif_1.append(*(Rot_1.apply(y_motif_1_tc[m])+y_motif_1_ini[k]))

            #no rotation
            if y_motif_1_rot==False:
                pos_list_y_motif_1.append(*(y_motif_1_tc[m]+y_motif_1_ini[k]))


        #pos_list_y_motif_1=np.asarray(Rot.random().apply(pos_list_y_motif_1))
        pos_list_y_motif_1=np.asarray(pos_list_y_motif_1)
           
        if use_y_1_HP_template==False:
            y_motif_1=simulation.add_topology("Particle",template_y_motif_1,pos_list_y_motif_1)
        if use_y_1_HP_template==True:
            y_motif_1=simulation.add_topology("Particle",template_y_motif_1_HP,pos_list_y_motif_1)

        conn_y_motif_1=np.arange(0,len(template_y_motif_1))
        #add graph for y-motif-1
        for i in range(3): #3 arms
            central_monomer=conn_y_motif_1[0]
            #add 3 harmonic polymer bonds in each arm
            for j in range(3):  #3 bonds per arm
                m=i*3
                k=j+m
                #each arm starts with central monomer
                if j==0:
                    arm_mon_a=central_monomer
                else:
                    arm_mon_a=conn_y_motif_1[k]

                arm_mon_b=conn_y_motif_1[k+1]
                #only backbone
                y_motif_1.get_graph().add_edge(arm_mon_a, arm_mon_b) 



    #add x-motif-1
    x_motif_1_tc=x_motif_template_coords(2.72)
    for k in range(num_x_motif_1):
        #x_motif_1_ini= np.random.uniform(size=(1,3)) * np.asarray([120,120,120]) - np.asarray([120,120,120]) * 0.5
        #x_motif_1_ini=random_spherical_vol(1,15)
        #x_motif_1_ini=np.asarray([[0,0,0]])
        pos_list_x_motif_1=[]
        if x_motif_1_rot==True:
            Rot_1=Rot.random()
        for m in range(len(x_motif_1_tc)):
            if x_motif_1_rot==True:
                pos_list_x_motif_1.append(*(Rot_1.apply(x_motif_1_tc[m])+x_motif_1_ini[k]))
            if x_motif_1_rot==False:
                pos_list_x_motif_1.append(*(x_motif_1_tc[m]+x_motif_1_ini[k]))

        pos_list_x_motif_1=np.asarray(pos_list_x_motif_1)

        if use_x_1_HP_template==False:
            x_motif_1=simulation.add_topology("Particle",template_x_motif_1,pos_list_x_motif_1)

        if use_x_1_HP_template==True:
            x_motif_1=simulation.add_topology("Particle",template_x_motif_1_HP,pos_list_x_motif_1)

        conn_x_motif_1=np.arange(0,len(template_x_motif_1))
        #add graph for y-motif-1
        for i in range(4): #4 arms
            central_monomer=conn_x_motif_1[0]
            #add 3 harmonic polymer bonds in each arm
            for j in range(3):  #3 bonds per arm
                m=i*3
                k=j+m
                #each arm starts with central monomer
                if j==0:
                    arm_mon_a=central_monomer
                else:
                    arm_mon_a=conn_x_motif_1[k]

                arm_mon_b=conn_x_motif_1[k+1]
                #only backbone
                x_motif_1.get_graph().add_edge(arm_mon_a, arm_mon_b)   

    #add x-motif-2
    x_motif_2_tc=x_motif_template_coords(2.72)
    for k in range(num_x_motif_2):
        #x_motif_2_ini= np.random.uniform(size=(1,3)) * np.asarray([120,120,120]) - np.asarray([120,120,120]) * 0.5
        #x_motif_2_ini=random_spherical_vol(1,15)
        #x_motif_2_ini=np.asarray([[0,0,0]])
        pos_list_x_motif_2=[]
        if x_motif_2_rot==True:
            Rot_2=Rot.random()
        for m in range(len(x_motif_2_tc)):
            if x_motif_2_rot==True:
                pos_list_x_motif_2.append(*(Rot_2.apply(x_motif_2_tc[m])+x_motif_2_ini[k]))
            if x_motif_2_rot==False:
                pos_list_x_motif_2.append(*(x_motif_2_tc[m]+x_motif_2_ini[k]))

        pos_list_x_motif_2=np.asarray(pos_list_x_motif_2)

        if use_x_2_HP_template==False:
            x_motif_2=simulation.add_topology("Particle",template_x_motif_2,pos_list_x_motif_2)

        if use_x_2_HP_template==True:
            x_motif_2=simulation.add_topology("Particle",template_x_motif_2_HP,pos_list_x_motif_2)
        conn_x_motif_2=np.arange(0,len(template_x_motif_2))
        #add graph for y-motif-1
        for i in range(4): #4 arms
            central_monomer=conn_x_motif_2[0]
            #add 3 harmonic polymer bonds in each arm
            for j in range(3):  #3 bonds per arm
                m=i*3
                k=j+m
                #each arm starts with central monomer
                if j==0:
                    arm_mon_a=central_monomer
                else:
                    arm_mon_a=conn_x_motif_2[k]

                arm_mon_b=conn_x_motif_2[k+1]
                #only backbone
                x_motif_2.get_graph().add_edge(arm_mon_a, arm_mon_b)  


    #add surface 1
    for i in range(num_surface_1):

        #surf_1_ini= np.random.uniform(size=(1,3)) * np.asarray([120,120,120]) - np.asarray([120,120,120]) * 0.5
        #surf_1_ini=[(0.1,0.1,0.1)]
        pos_list_surf=surf_template_coords(2.72)+np.asarray(surf_1_ini[i])


        pol=simulation.add_topology("Particle",template_surface_1,pos_list_surf)

        for k in range(len(template_surface_1)-1):
            pol.get_graph().add_edge(k, k+1) 
##


###########
#start ReaDDy sim
###########

#run simulation          
def run_sim_1(name,
             folder,
             add_num_1,
              
             system_temperature,
             box_size,
             custom_units,
             periodic_bc_x,
             periodic_bc_y,
             periodic_bc_z,
              
             r_sp,
             sp_incl_orig,
             k_sp,
             box_a,
             box_b,
             k_box,
    
              
             num_y_motif_1,
             num_x_motif_1,
             num_x_motif_2,
             num_surface_1,
             y_motif_1_ini,
             x_motif_1_ini,
             x_motif_2_ini,
             surf_1_ini,
             y_motif_1_rot,
             x_motif_1_rot,
             x_motif_2_rot,
              
             use_y_1_HP_template,
             use_x_1_HP_template,
             use_x_2_HP_template,
             
             Diff_sys_temp,
             k_bond_ds,
             k_bond_ss,
             eq_dist_bb,
             eq_dist_fused_bb,
             eq_dist_bb_s,
             k_angle_ds_base,
             k_angle_ds_arm,
             k_angle_ds_oppa,
             k_angle_ss_link,
             k_angle_ds_fused_link,
             k_angle_ds_hp,
             k_angle_ss_surf,
             k_angle_ds_surf,
             theta_0,
             theta_90,
             theta_60,
             theta_120,
             theta_270,
             theta_base_x,
             theta_base_y,
             theta_oppa_x,
             
             volume_excl_type,
             k_ex_ds_ds,
             k_ex_ss_ds,
             k_ex_ss_ss,
             dist_ex,
             DH_strength_ds_ds,
             DH_strength_ss_ds,
             DH_strength_ss_ss,
             DH_len,
             DH_cutoff,
              
              
             k_fiss_rate_sel,
             k_b_y_1,
             k_ub_y_1,
             k_b_y_hp_1,
             k_ub_y_hp_1,
             k_b_x_1,
             k_ub_x_1,
             k_b_x_surf_1,
             k_ub_x_surf_1,
             k_b_x_hp_1,
             k_ub_x_hp_1,
             k_b_x_surf_hp_1,
             k_ub_x_surf_hp_1,
             k_b_x_2,
             k_ub_x_2,
             k_b_x_surf_2,
             k_ub_x_surf_2,
             k_b_x_hp_2,
             k_ub_x_hp_2,
             k_b_x_surf_hp_2,
             k_ub_x_surf_hp_2,
                
             t_step,
             rea_s,
             t_tot,
             
             sim_kernel,
             sim_threads,
             track_num_particles,
             eval_reac,
             eval_top_reac,
             checkpoint_path,
             new_checkpoint,
             max_n_saves,
             rec_stride, 
              sim_skin, 
             
            ):
    
    #########################################
    #########################################
    #initialize system
    system = readdy.ReactionDiffusionSystem(box_size=box_size,unit_system=custom_units)
    system.periodic_boundary_conditions =periodic_bc_x,periodic_bc_y,periodic_bc_z 
    system.temperature=system_temperature
    system.topologies.add_type("Particle")

    #basic parameters and potentials
    for i in all_types:
        #add monomer
        #print(i)
        system.add_topology_species(i,Diff_sys_temp)
        #spherical inclusion
        if k_sp!=0:
            system.potentials.add_sphere(i,k_sp,sp_incl_orig, r_sp,True)

        #box potential
        if k_box!=0:
            system.potentials.add_box(i, k_box, box_a,box_b)

    #########################################
    #########################################
    #Y-MOTIF

    add_y_motif_potentials(system,list_base_types_y_motif,list_arm_types_y_motif,list_link_types_y_motif,list_link_fused_types_y_motif,list_link_hp_types_y_motif_1,
                       
                           k_bond_ds,k_bond_ss,eq_dist_bb,eq_dist_bb_s,eq_dist_fused_bb,
                       
                           k_angle_ds_base,k_angle_ds_arm,k_angle_ss_link,k_angle_ds_hp,k_angle_ds_fused_link,
                       
                           theta_base_y,theta_0)

    #########################################
    #########################################
    #X-MOTIF
    #x1 and x2 with same parameters
    add_x_motif_potentials(system,list_base_types_x_motif,list_arm_types_x_motif,list_link_types_x_motif_1,list_link_types_x_motif_2,
                           
                           list_link_fused_types_x_motif_1,list_link_fused_types_x_motif_2,
                           
                           list_link_hp_types_x_motif_1,list_link_hp_types_x_motif_2,
                           
                           k_bond_ds,k_bond_ss,eq_dist_bb,eq_dist_bb_s,eq_dist_fused_bb,
                           
                           k_angle_ds_base,k_angle_ds_arm,k_angle_ss_link,k_angle_ds_hp,k_angle_ds_fused_link,k_angle_ds_oppa,
                           
                           theta_base_x,theta_0,theta_oppa_x) 
   
    #########################################
    #########################################
    #SURFACE
    #potentials in surface
    add_surface_potentials(system,list_surface_type_1,list_surface_type_1_ss,list_surface_type_1_ds,
                          
                           k_bond_ds,k_bond_ss,eq_dist_bb,
                           
                           k_angle_ss_surf,k_angle_ds_surf,
                           
                           theta_0)

    #potentials surface-xmotifs
    system.topologies.configure_harmonic_bond("x_surf_bound_int_1","x_surf_link_fused_d_1", force_constant=k_bond_ds, length=eq_dist_bb_s)
    system.topologies.configure_harmonic_bond("x_surf_bound_int_1","x_surf_link_fused_d_2", force_constant=k_bond_ds, length=eq_dist_bb_s)
    #do we need angle bound sticky end surface??

    #########################################
    #########################################
    #REPULSION
    #harmonic or DH type particle-particle interaction
    add_interaction_potentials(system,all_types,list_ss_DNA_particles_1,
                               
                               volume_excl_type,
                              
                               k_ex_ss_ds,k_ex_ss_ss,k_ex_ds_ds,
                               
                               DH_strength_ss_ds,DH_strength_ss_ss,DH_strength_ds_ds,DH_len,DH_cutoff)
                
    #########################################
    #########################################
    #REACTIONS
    
    #pre loading reaction params fission sel
    rate_fission_selection=partial(param_rate_fission_selection, k=k_fiss_rate_sel)
    
    #calc exp rates for fission
    
    exp_rate_ub_y_1=1-np.exp(-t_step * k_ub_y_1) #y link fission
    exp_rate_ub_x_1=1-np.exp(-t_step * k_ub_x_1) #x link 1 fission
    exp_rate_ub_x_surf_1=1-np.exp(-t_step * k_ub_x_surf_1) #x link 1 surf fission
    exp_rate_ub_x_2=1-np.exp(-t_step * k_ub_x_2) #x link 2 fission
    exp_rate_ub_x_surf_2=1-np.exp(-t_step * k_ub_x_surf_2) #x link 2 surf fission

    #pre loading reaction params fission reaction
    fission_5=partial(param_fission_5,exp_rate_ub_y_1=exp_rate_ub_y_1,exp_rate_ub_x_1=exp_rate_ub_x_1,exp_rate_ub_x_surf_1=exp_rate_ub_x_surf_1,exp_rate_ub_x_2=exp_rate_ub_x_2,exp_rate_ub_x_surf_2=exp_rate_ub_x_surf_2,
                     list_link_fused_types_y_motif=list_link_fused_types_y_motif,
                      list_link_fused_types_x_motif_1=list_link_fused_types_x_motif_1,
                      list_link_fused_types_x_motif_2=list_link_fused_types_x_motif_2)

    #fusion reaction y_link_1  
    #particles with y_link_1 fuse
    #account for all types of linkers in y motif
    gen_fusion_reactions_1(fusion_input_particles=list_link_types_y_motif,fusion_output_particles=list_link_fused_types_y_motif,rate=k_b_y_1,distance=rea_s,system=system)

    #fusion reaction x_link_1
    #account for all types of linkers in x motif (except)
    #use list_link_types_x_motif_1[:3] to exclude surface binding sticky ends
    gen_fusion_reactions_1(fusion_input_particles=list_link_types_x_motif_1[:3],fusion_output_particles=list_link_fused_types_x_motif_1[:3],rate=k_b_x_1,distance=rea_s,system=system)

    #fusion reaction x_link_2
    gen_fusion_reactions_1(fusion_input_particles=list_link_types_x_motif_2[:3],fusion_output_particles=list_link_fused_types_x_motif_2[:3],rate=k_b_x_2,distance=rea_s,system=system)

    #fusion x_link_1 and surface
    fusion_name_x_link_surf_1="X_Surf_Fusion_1: Particle(x_surf_link_d_1) + Particle(x_surf_int_1) -> Particle(x_surf_link_fused_d_1--x_surf_bound_int_1)"
    system.topologies.add_spatial_reaction(fusion_name_x_link_surf_1, rate=k_b_x_surf_1, radius=rea_s) 

    #fusion x_link_2 and surface
    fusion_name_x_link_surf_2="X_Surf_Fusion_2: Particle(x_surf_link_d_2) + Particle(x_surf_int_1) -> Particle(x_surf_link_fused_d_2--x_surf_bound_int_1)"
    system.topologies.add_spatial_reaction(fusion_name_x_link_surf_2, rate=k_b_x_surf_2, radius=rea_s) 


    #fission reaction 
    system.topologies.add_structural_reaction("Fission_1","Particle",fission_5, rate_fission_selection)

    #Hairpin formation reaction
    #y motif 
    system.reactions.add("HP_formation_y_1_a: y_link_a_1 -> y_link_hp_a_1",rate=k_b_y_hp_1)
    system.reactions.add("HP_formation_y_1_b: y_link_b_1 -> y_link_hp_b_1",rate=k_b_y_hp_1)
    system.reactions.add("HP_formation_y_1_c: y_link_c_1 -> y_link_hp_c_1",rate=k_b_y_hp_1)
    #################
    system.reactions.add("HP_breaking_y_1_a: y_link_hp_a_1 -> y_link_a_1",rate=k_ub_y_hp_1)
    system.reactions.add("HP_breaking_y_1_b: y_link_hp_b_1 -> y_link_b_1",rate=k_ub_y_hp_1)
    system.reactions.add("HP_breaking_y_1_c: y_link_hp_c_1 -> y_link_c_1",rate=k_ub_y_hp_1)
    
    #x motif 1+2
    system.reactions.add("HP_formation_x_1_a: x_link_a_1 -> x_link_hp_a_1",rate=k_b_x_hp_1)
    system.reactions.add("HP_formation_x_1_b: x_link_b_1 -> x_link_hp_b_1",rate=k_b_x_hp_1)
    system.reactions.add("HP_formation_x_1_c: x_link_c_1 -> x_link_hp_c_1",rate=k_b_x_hp_1)
    system.reactions.add("HP_formation_x_1_d: x_surf_link_d_1 -> x_surf_link_hp_d_1",rate=k_b_x_surf_hp_1)
    #
    system.reactions.add("HP_formation_x_2_a: x_link_a_2 -> x_link_hp_a_2",rate=k_b_x_hp_2)
    system.reactions.add("HP_formation_x_2_b: x_link_b_2 -> x_link_hp_b_2",rate=k_b_x_hp_2)
    system.reactions.add("HP_formation_x_2_c: x_link_c_2 -> x_link_hp_c_2",rate=k_b_x_hp_2)
    system.reactions.add("HP_formation_x_2_d: x_surf_link_d_2 -> x_surf_link_hp_d_2",rate=k_b_x_surf_hp_2)
    ###############
    system.reactions.add("HP_breaking_x_1_a: x_link_hp_a_1 -> x_link_a_1",rate=k_ub_x_hp_1)
    system.reactions.add("HP_breaking_x_1_b: x_link_hp_b_1 -> x_link_b_1",rate=k_ub_x_hp_1)
    system.reactions.add("HP_breaking_x_1_c: x_link_hp_c_1 -> x_link_c_1",rate=k_ub_x_hp_1)
    system.reactions.add("HP_breaking_x_1_d: x_surf_link_hp_d_1 -> x_surf_link_d_1",rate=k_ub_x_surf_hp_1)
    #
    system.reactions.add("HP_breaking_x_2_a: x_link_hp_a_2 -> x_link_a_2",rate=k_ub_x_hp_2)
    system.reactions.add("HP_breaking_x_2_b: x_link_hp_b_2 -> x_link_b_2",rate=k_ub_x_hp_2)
    system.reactions.add("HP_breaking_x_2_c: x_link_hp_c_2 -> x_link_c_2",rate=k_ub_x_hp_2)
    system.reactions.add("HP_breaking_x_2_d: x_surf_link_hp_d_2 -> x_surf_link_d_2",rate=k_ub_x_surf_hp_2)


    simulation = system.simulation(kernel=sim_kernel)
    #simulation = system.simulation(kernel="CPU")
    if sim_kernel=="CPU":
        simulation.kernel_configuration.n_threads = sim_threads
    #simulation.reaction_handler = "UncontrolledApproximation"



  
    #########################################
    ########################################
    
    #ADD PARTICLES
    
    if checkpoint_path==None:
        
        add_particles_to_sim(simulation,num_y_motif_1,y_motif_1_ini,y_motif_1_rot,template_y_motif_1,template_y_motif_1_HP,y_motif_template_coords,
                        
                         num_x_motif_1,x_motif_1_ini,x_motif_1_rot,template_x_motif_1,template_x_motif_1_HP, 
                         
                         num_x_motif_2,x_motif_2_ini,x_motif_2_rot,template_x_motif_2,template_x_motif_2_HP, x_motif_template_coords,
                         
                         use_y_1_HP_template,use_x_1_HP_template,use_x_2_HP_template,
                         
                      num_surface_1,surf_1_ini,template_surface_1,surf_template_coords)
               
    else:
        simulation.load_particles_from_latest_checkpoint(checkpoint_path)
        
    #########################################
    #########################################
    #Record output

    output_file_name=folder+name+str(int(add_num_1))+".h5"
    rec_stride=rec_stride
    simulation.output_file = output_file_name
    simulation.record_trajectory(rec_stride)        
    simulation.observe.particles(rec_stride)
    simulation.observe.energy(stride=rec_stride)
    simulation.observe.topologies(rec_stride)
    simulation.observe.reaction_counts(rec_stride)
    #simulation.observe.number_of_particles(stride=rec_stride,types=["y_link_1","y_link_fused_1","star_link_1","star_link_fused_1","star_y_link_1","star_y_link_fused_1"])
    #simulation.observe.number_of_particles(stride=rec_stride,types=["y_link_1","y_link_fused_1"])
    simulation.observe.number_of_particles(stride=rec_stride,types=track_num_particles)

    simulation.progress_output_stride = 1

    simulation.evaluate_topology_reactions = eval_top_reac
    simulation.evaluate_reactions = eval_reac #does not work??

    simulation.skin =sim_skin
    
    if new_checkpoint!=None:
        simulation.make_checkpoints(rec_stride, output_directory=new_checkpoint, max_n_saves=max_n_saves)


    simulation.run(t_tot, t_step )






############
#Tcl file for y,x-motif and surface
############

def tcl_file_gen(traj_file,box_dim_s):
                
    radii={}
    for i in range(len(all_types)):
        radii[all_types[i]]=1.36
        
    
    tcl_file=[]   #'y_centre_1', 'y_arm_1', 'y_link_1', 'y_link_fused_1', 'star_centre_1', 'star_arm_1', 'star_link_1', 'star_y_link_1', 'neg_charge_1', 'star_link_fused_1', 'star_y_link_fused_1'
    tcl_file.append("mol delete top")
    name_load="mol load xyz "+traj_file
    tcl_file.append(name_load)
    tcl_file.append("mol delrep 0 top")
    tcl_file.append("display resetview")
    #add monomers
    for i in range(len(all_types)):
        
        tcl_file.append("mol representation VDW 0.952000 16.0")
        
        ### y-motif ###
        if all_types[i] in ["y_arm_1",*list_base_types_y_motif,*list_arm_types_y_motif]:  #red
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 1")
                            
        if all_types[i]=="y_centre_1": #pink
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 9")
                            
        if all_types[i] in ["y_link_1", *list_link_types_y_motif ]: #purple
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 11")
                            
        if all_types[i] in ["y_link_fused_1",*list_link_fused_types_y_motif]: # orange
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 3")
        
        
        ###############
        
        ### x-motif ### 
        if all_types[i] in ["x_arm_1",*list_base_types_x_motif,*list_arm_types_x_motif]:  #red
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 1")
                            
        if all_types[i]=="x_centre_1": #pink
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 9")
        #motif 1                   
        if all_types[i] in [*list_link_types_x_motif_1[:3]]: #purple
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 11")
        #motif 2 
        if all_types[i] in [*list_link_types_x_motif_2[:3] ]: #silver
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 6")
        #motif 1   
        if all_types[i] in [*list_link_fused_types_x_motif_1[:3]]: # orange
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 3")
        #motif 2
        if all_types[i] in [*list_link_fused_types_x_motif_2[:3]]: # yellow
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 4")
            
        if all_types[i] in [list_link_types_x_motif_1[3],list_link_types_x_motif_2[3]]: # cyan
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 10")
            
            
        if all_types[i] in [list_link_fused_types_x_motif_1[3],list_link_fused_types_x_motif_2[3]]: # orange 3
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 32")  
            
        #motif 1    
        if all_types[i] in list_link_hp_types_x_motif_1: # lime
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 12") 
            
        #motif 2   
        if all_types[i] in list_link_hp_types_x_motif_2: # tan
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 5")    
            
        #########
        
        ### surface ###
        if all_types[i]=="x_surf_fill_1": #black
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 16")
                            
        if all_types[i]=="x_surf_int_1": #blue
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 0")
                            
        if all_types[i]=="x_surf_bound_int_1": # orange 3
            ins="mol selection name "+"type_"+str(i)
            tcl_file.append(ins)
            tcl_file.append("mol color ColorID 32")
      
            
        tcl_file.append("mol addrep top")
    


    tcl_file.append("animate goto 0")
    tcl_file.append("color Display Background white")
    tcl_file.append("molinfo top set {center_matrix} {{{1 0 0 0}{0 1 0 0}{0 0 1 0}{0 0 0 1}}}")
    box_s="set cell [pbc set {"+box_dim_s+"} -all]"
    tcl_file.append(box_s)
    tcl_file.append("pbc box -center origin -color black -width 1")
    
    return tcl_file

##
