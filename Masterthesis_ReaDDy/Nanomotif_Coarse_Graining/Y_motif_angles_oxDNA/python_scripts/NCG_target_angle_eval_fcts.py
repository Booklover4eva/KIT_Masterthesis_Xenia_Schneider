#functions used in NCG_target_angle_eval_x_motif notebook

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
print(readdy.__version__)
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import Counter
import random

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
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d

from multiprocessing import Process, Manager


#read coords. from oxDNA simulation
def read_positions_2(file_name,num_nucl,time_start,time_stop,time_step):
    #file_name, path to trajectory.dat file from oxDNA
    #num_nucl= how many nucleotides where simulated
    #time_start= time step when to start sampling
    #time_stop= time step wehn to stop sampling
    #time_step= take every (=1) or every nth (=n) time step for sampling
    file=pd.read_csv(file_name, skiprows=3, usecols=[0,1,2], header=None,delim_whitespace=True)

    file = file[~file[0].isin(["t","b","E"])] #.to_numpy()
    
    pos_all=[]
    n_time_points=(time_stop-time_start)/time_step
    for i in range(int(n_time_points)):
        k=i*time_step #e.g. 0,5,10,15,...
        start_row=num_nucl*k +time_start*num_nucl

        pos_at_t=file.iloc[start_row:start_row+num_nucl,:].to_numpy(dtype='float')
        #print(pos_at_t,start_row,start_row+num_nucl)
        pos_all.append(pos_at_t)
        print(i+1,"/",int(n_time_points))
    return np.asarray(pos_all)


#angle between two vectors 
def angle_1(v1,v2):
    v1_n=v1/np.linalg.norm(v1)
    v2_n=v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_n,v2_n),-1,1))
    
#centre of mass of list of coords.
def cent_of_mass(list_of_coords):
    #list_of_coords= [[x1,y1,z1], [x2,y2,z2],...]
    x, y, z = np.array(list_of_coords).T
    cent = [np.mean(x), np.mean(y), np.mean(z)]
    return np.asarray(cent)
    
#get angles in nanomotif by defining groups of nucleotides which will be used to calculate center of mass and vectors between centres of mass
def conf_angles_3(pos, arm_vector_indices):
    #pos=list with [time][nucleotide][x,y,z] 
    #arm_vector_indices= [arm i][tip1,base1,tip2,base2][index 1, ...index n]

    
    #angles between arms
    angles_arms=[]
    #angles between arms defined by PCA vectors
    #angles_arms_PCA=[]

    #get all vectors for time point i
    #use indices to get centres of mass, use centres of mass to get vectors
    for i in range(len(pos)):#iterate time points
        
        angle_arm_n=[] #list with [angle_arm_1,...,angle_arm_n]
        for n in range(len(arm_vector_indices)): #iterate arms
            
            #vec_tip_base_n=[] #list containing [vector_1_arm_n,vector_2_arm_n]
            cent_of_mass_tip_base_n=[]# list containing [centre_mass_tip1,centre_mass_base_1,centre_mass_tip2,centre_mass_base_2]  for arm n
            for m in range(len(arm_vector_indices[n])): #iterate tips, bases in each arm

                coords_tip_base_n=[]
                #print("#",len(arm_vector_indices[n][m]))
                for k in range(len(arm_vector_indices[n][m])): #iterate indices for tip or base n
                    index_tip_base_n=arm_vector_indices[n][m][k]
                    #print(index_tip_base_n)
                    coords_tip_base_n.append(pos[i][index_tip_base_n])
                    
                #get centre of mass for tip or base n
                #print(coords_tip_base_n,cent_of_mass(list_of_coords=coords_tip_base_n))   
                #print(coords_tip_base_n)
                cent_of_mass_tip_base_n.append(cent_of_mass(list_of_coords=coords_tip_base_n))
            #get vectors between tip 1 base 1 and tip 2 base 2 of arm n
            vector_1_arm_n=cent_of_mass_tip_base_n[0]-cent_of_mass_tip_base_n[1]
            vector_2_arm_n=cent_of_mass_tip_base_n[2]-cent_of_mass_tip_base_n[3]
            
            #get angle between vector 1 and vector 2
            vector_1_2_arm_n=angle_1(vector_1_arm_n,vector_2_arm_n)
            angle_arm_n.append(vector_1_2_arm_n)
        #append 
        print(i+1,"/",len(pos))
        angles_arms.append(angle_arm_n)
    return np.asarray(angles_arms)
    
    
#https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249

#calculate angles between vectors aligned to each arm instead of arm-centre
#use correct split in arms pos
#additionally use PCA and orient vectors in PCA to be aligned towards ends of arms
def conf_angles_2c(pos,arm_indices, split_arm_end_start):
    #pos=list with [time][nucleotide][x,y,z] 
    #arm_indices= [arm i][index 1, ...index n]
    #centre indices= [index 1, ... index k]
    #split_arm_end_start= index where to split points in arms between points belonging to end of arm vs points belonging to start of arm, if 10 points per arm, split_arm_end_start=5 for symmetry

    
    #angles between arms
    angles_arms=[]
    #angles between arms defined by PCA vectors
    angles_arms_PCA=[]

    #get all vectors for time point i
    #use indices to get centres of mass, use centres of mass to get vectors
    for i in range(len(pos)):#iterate time points
        
        #vectors: vec(arm i) - vec(centre)
        list_of_vectors=[]
        #vectors defined by PCA, orientation along vectors defined above
        list_of_vectors_PCA=[]
        
        #### get centre of masses for start and end of all arms in motif####
        cent_mass_arms_start=[]
        cent_mass_arms_end=[]

        #also get all coordinates in each arm for PCA
        coords_arm_PCA=[]
        for n in range(len(arm_indices)):#iterate arms
            
            #get coordinates of all nucleotides to be used for centre of mass
            pos_at_time_in_arm_start=[]
            pos_at_time_in_arm_end=[]

            for p in range(len(arm_indices[n][0])): #3-5 arm 
                index_arm=arm_indices[n][0][p]
                #print(index_arm)
                if p<split_arm_end_start:
                    pos_at_time_in_arm_start.append(pos[i][index_arm])
                if p>=split_arm_end_start:
                    pos_at_time_in_arm_end.append(pos[i][index_arm])

                      

            for p in range(len(arm_indices[n][1])): #5-3 arm
                index_arm=arm_indices[n][1][p]
                #print(index_arm)
                if p<split_arm_end_start:
                    pos_at_time_in_arm_start.append(pos[i][index_arm])
                if p>=split_arm_end_start:
                    pos_at_time_in_arm_end.append(pos[i][index_arm])

            
            #get centre of mass               
            cent_mass_arms_start.append(cent_of_mass(list_of_coords=pos_at_time_in_arm_start))
            cent_mass_arms_end.append(cent_of_mass(list_of_coords=pos_at_time_in_arm_end))

            #get all points for additional PCA
            
            coords_arm_PCA.append([*pos_at_time_in_arm_start,*pos_at_time_in_arm_end] )
            
        
        
        #print(cent_mass_arms)
        #calculate vectors from centre to arm centres of masses
        for s in range(len(cent_mass_arms_start)):
            list_of_vectors.append(cent_mass_arms_start[s]-cent_mass_arms_end[s])
        #print(list_of_vectors)
        #print(cent_mass_centre)

        for b in range(len(coords_arm_PCA)):
            pca_e=PCA(n_components=1).fit(coords_arm_PCA[b]).components_[0]
            if np.dot(pca_e,list_of_vectors[b])<0: #check if PCA vector was chosen in right orientation
                
                list_of_vectors_PCA.append( -pca_e )
            else:
                list_of_vectors_PCA.append( pca_e )
            
        angles_arms_t=[]
        angles_arms_t_PCA=[]
        for z in range(len(list_of_vectors)):
            ind1=z
            ind2=(z+1)%len(list_of_vectors) #e.g. 0-1, 1-2, 2-0
            #print(ind1,ind2)
            angles_arms_t.append(angle_1(list_of_vectors[ind1],list_of_vectors[ind2]))
            angles_arms_t_PCA.append(angle_1(list_of_vectors_PCA[ind1],list_of_vectors_PCA[ind2]))
            
        angles_arms.append(angles_arms_t)
        angles_arms_PCA.append(angles_arms_t_PCA)
        #calculate angles between arm and vector that is sum of two other arms (i.e. in plane formed by those two vectors)

        print(i+1,"/",len(pos))
    return np.asarray(angles_arms),np.asarray(angles_arms_PCA)
 
 
#calculate angles between vectors aligned to each arm instead of arm-centre
#use correct split in arms pos
#additionally use PCA and orient vectors in PCA to be aligned towards ends of arms
#based on conf_angles_2c, but for angle between opposing arms
def conf_angles_2d(pos,arm_indices, split_arm_end_start):
    #pos=list with [time][nucleotide][x,y,z] 
    #arm_indices= [arm i][index 1, ...index n]
    #centre indices= [index 1, ... index k]
    #split_arm_end_start= index where to split points in arms between points belonging to end of arm vs points belonging to start of arm, if 10 points per arm, split_arm_end_start=5 for symmetry

    
    #angles between arms
    angles_arms=[]
    #angles between arms defined by PCA vectors
    angles_arms_PCA=[]

    #get all vectors for time point i
    #use indices to get centres of mass, use centres of mass to get vectors
    for i in range(len(pos)):#iterate time points
        
        #vectors: vec(arm i) - vec(centre)
        list_of_vectors=[]
        #vectors defined by PCA, orientation along vectors defined above
        list_of_vectors_PCA=[]
        
        #### get centre of masses for start and end of all arms in motif####
        cent_mass_arms_start=[]
        cent_mass_arms_end=[]

        #also get all coordinates in each arm for PCA
        coords_arm_PCA=[]
        for n in range(len(arm_indices)):#iterate arms
            
            #get coordinates of all nucleotides to be used for centre of mass
            pos_at_time_in_arm_start=[]
            pos_at_time_in_arm_end=[]

            for p in range(len(arm_indices[n][0])): #3-5 arm 
                index_arm=arm_indices[n][0][p]
                #print(index_arm)
                if p<split_arm_end_start:
                    pos_at_time_in_arm_start.append(pos[i][index_arm])
                if p>=split_arm_end_start:
                    pos_at_time_in_arm_end.append(pos[i][index_arm])

                      

            for p in range(len(arm_indices[n][1])): #5-3 arm
                index_arm=arm_indices[n][1][p]
                #print(index_arm)
                if p<split_arm_end_start:
                    pos_at_time_in_arm_start.append(pos[i][index_arm])
                if p>=split_arm_end_start:
                    pos_at_time_in_arm_end.append(pos[i][index_arm])

            
            #get centre of mass               
            cent_mass_arms_start.append(cent_of_mass(list_of_coords=pos_at_time_in_arm_start))
            cent_mass_arms_end.append(cent_of_mass(list_of_coords=pos_at_time_in_arm_end))

            #get all points for additional PCA
            
            coords_arm_PCA.append([*pos_at_time_in_arm_start,*pos_at_time_in_arm_end] )
            
        
        
        #print(cent_mass_arms)
        #calculate vectors from centre to arm centres of masses
        for s in range(len(cent_mass_arms_start)):
            list_of_vectors.append(cent_mass_arms_start[s]-cent_mass_arms_end[s])
        #print(list_of_vectors)
        #print(cent_mass_centre)

        for b in range(len(coords_arm_PCA)):
            pca_e=PCA(n_components=1).fit(coords_arm_PCA[b]).components_[0]
            if np.dot(pca_e,list_of_vectors[b])<0: #check if PCA vector was chosen in right orientation
                
                list_of_vectors_PCA.append( -pca_e )
            else:
                list_of_vectors_PCA.append( pca_e )
            
        angles_arms_t=[]
        angles_arms_t_PCA=[]

        for z in range(int(len(list_of_vectors)/2)):
            ind1=z
            ind2=z+int(len(list_of_vectors)/2) #e.g. 0-2, 1-3
            angles_arms_t.append(angle_1(list_of_vectors[ind1],list_of_vectors[ind2]))
            angles_arms_t_PCA.append(angle_1(list_of_vectors_PCA[ind1],list_of_vectors_PCA[ind2]))  
            
        #angle arm 0-2:
        #angles_arms_t.append(angle_1(list_of_vectors[0],list_of_vectors[2]))
        #angles_arms_t_PCA.append(angle_1(list_of_vectors_PCA[0],list_of_vectors_PCA[2]))

        #angle arm 1-3:
        #angles_arms_t.append(angle_1(list_of_vectors[1],list_of_vectors[3]))
        #angles_arms_t_PCA.append(angle_1(list_of_vectors_PCA[1],list_of_vectors_PCA[3]))
        
        angles_arms.append(angles_arms_t)
        angles_arms_PCA.append(angles_arms_t_PCA)

        print(i+1,"/",len(pos))
    return np.asarray(angles_arms),np.asarray(angles_arms_PCA)

