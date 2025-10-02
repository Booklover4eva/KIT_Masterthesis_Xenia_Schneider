# Functions for evaluation of readdy simulations of nanomotifs
#Version 2: based on Version 1, with updated methods to calculate complex moduli, based on using connected components in graph for separate G' and G'' curves
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import readdy
import math
import scipy
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import contextlib
import io
import pickle
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import multiprocessing

import threading

import sys
import os
from collections import defaultdict, deque

import contextlib
import io
import pickle



#####################
# General data
#####################

#get number of particles as a function of time from simulations
def get_particle_number_from_sims(folder,name,add_num,index_particle_tpye):
    part_number=[]
    for i in range(len(add_num)):
        #part_number_e=[]
        name_load_in=folder+name+str(add_num[i])+".h5"
        
        traj=readdy.Trajectory(name_load_in)

        times,types,ids,part_positions=traj.read_observable_particles()
        time_pt, counts_pt = traj.read_observable_number_of_particles()
        
        #for t in range(len(time_pt)):
            #count=0
        count=[]
        for j in range(len(index_particle_tpye)):
            count.append(counts_pt[:,index_particle_tpye[j]] )
            
        part_number.append(np.sum(count,axis=0) )
        #part_number.append(part_number_e)
    return np.asarray(part_number)
    
    
######################    
#### Rheological data
######################

#Theory described in Cohen et al 2024, Direct computation of viscoelastic moduli of biomolecular condensates



#get list of edges between all particles
#replace indices of edges ranging from 0 to topology length with ids of particles in topology for certain time step
def get_list_of_all_edges(time_sel,tops):
    edges_list_1=[]
    for top in tops[time_sel]:
        particles=top.particles
        edges=top.edges

        edges_with_ids=[]
        for k in range(len(edges)):
            #get indices of edges from 0 to length of top
            index_a_edge=edges[k][0]
            index_b_edge=edges[k][1]
            #convert to indicies of edges using particle id
            index_id_a_edge=particles[index_a_edge]
            index_id_b_edge=particles[index_b_edge]
            edges_with_ids.append((index_id_a_edge,index_id_b_edge))

        edges_list_1.append(edges_with_ids)
    edges_list_1 = [e for sublist in edges_list_1 for e in sublist]
    return edges_list_1
    
    
#Get all indices of vertices with four edges= centre particle
#Starting from the index of each centre particle get all indices of vertices/particles three edges away=linker particles
#get the non-surface linkers
def vertices_with_n_edges(edges,count_edges):
    # Create a dictionary to count the number of edges for each vertex
    edge_count = defaultdict(int)
    
    # Iterate over each edge and update the count for both vertices
    for edge in edges:
        edge_count[edge[0]] += 1
        edge_count[edge[1]] += 1
    
    # Get the list of vertices that have exactly four edges
    result = [vertex for vertex, count in edge_count.items() if count == count_edges]
    
    return result

def vertices_n_edges_away(edges, start_vertex,number_edges):
    # Create an adjacency list for the graph
    adjacency_list = defaultdict(list)
    
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    
    # Perform BFS to find vertices that are exactly three edges away
    queue = deque([(start_vertex, 0)])
    visited = set([start_vertex])
    result = []
    
    while queue:
        current_vertex, distance = queue.popleft()
        
        if distance == number_edges:
            result.append(current_vertex)
        
        if distance < number_edges:
            for neighbor in adjacency_list[current_vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    return result
    
#create adjacency matrix
#start from list of lists with linker vertices [[id_linker_1_motif_1,id_linker_2_motif_1,id_linker_3_motif_1,id_linker_4_motif_1],]
def create_adjacency_matrix(edges, sublists):
    # Create an adjacency list for the graph
    adjacency_list = defaultdict(list)
    
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    
    # Initialize the adjacency matrix with zeros
    n = len(sublists)
    matrix = [[0] * n for _ in range(n)]
    
    # Check for edges between vertices of different sublists
    for i in range(n):
        for j in range(i + 1, n):
            for vertex_i in sublists[i]:
                for vertex_j in sublists[j]:
                    if vertex_j in adjacency_list[vertex_i]:
                        matrix[i][j] = 1
                        matrix[j][i] = 1
                        break
                if matrix[i][j] == 1:
                    break
    
    return matrix
    
    
    
    
#function that converts edges at time point t into connectivity matrix and eigenvalues from observed topology

def get_conn_mat_eigv_1(time_sel,obs_top,edges_centre,edges_linker):
    #time_sel: time point to evaluate
    #obs_top: observed topologies for all time steps
    #edges_centre: get vertices with this number of edges
    #edges_linker:get linkers this number of edges away from centre particles

    
    #all edges at time point time_sel
    edges_ids_t=get_list_of_all_edges(time_sel=time_sel,tops=obs_top)

    #get all vertices with n edges (i.e. all centre particles)
    vertices_with_n_edges_t=vertices_with_n_edges(edges=edges_ids_t,count_edges=edges_centre)

    #get all vertices m edges away from centre particles (i.e. get all linker particles)
    vertices_m_edges_away_t=[]
    for i in range(len(vertices_with_n_edges_t)):
        check_v=vertices_with_n_edges_t[i]
        vertices_m_edges_away_t.append( vertices_n_edges_away(edges=edges_ids_t, start_vertex=check_v,number_edges=edges_linker) )
        
        

    #create adjacency matrix 
    adjacency_matrix_t=create_adjacency_matrix(edges=edges_ids_t, sublists=vertices_m_edges_away_t)
    
    #create connectivity matrix 
    connectivity_matrix_t=np.zeros( (len(adjacency_matrix_t),len(adjacency_matrix_t))  )
    #fill main diagonal with number of edges (i.e. sum of each column)
    for i in range(len(connectivity_matrix_t)):
            connectivity_matrix_t[i][i]=np.sum(adjacency_matrix_t[i])
    #fill off diagonal entries with entries of connectivity matrix    
    for i in range(len(connectivity_matrix_t)):
        for j in range(i): #dont go to main diagonal 
            connectivity_matrix_t[i][j]=adjacency_matrix_t[i][j]*-1
            connectivity_matrix_t[j][i]=adjacency_matrix_t[i][j]*-1


    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(connectivity_matrix_t)

    return connectivity_matrix_t, eigenvalues
    
    
#get eigenvalues of graph laplacian split by connected sub-components (i.e. for each topology) ordered by decending component size 
def get_subgraph_eigenvalues(laplacian):
    #laplacian:graph laplacian with arbitrary labelling of nodes
    laplacian_sparse = csr_matrix(laplacian)
    n_components, labels = connected_components(csgraph=laplacian_sparse, directed=False, return_labels=True)

    components = []

    for component in range(n_components):
        component_indices = np.where(labels == component)[0]
        sub_laplacian = laplacian[np.ix_(component_indices, component_indices)]
        sub_eigenvalues = eigh(sub_laplacian, eigvals_only=True)
        components.append((len(component_indices), sub_eigenvalues))

    # Sort by component size (descending)
    components.sort(reverse=True, key=lambda x: x[0]) # components=[[size of component 0, [array of eigenvalues 0]],[size of component 1, [array of eigenvalues 1],...]

    return components
    
    
#function that converts edges at time point t into connectivity matrix and eigenvalues from observed topology
#Version 2: get eigenvalues of each connected sub component i.e. each topology 
def get_conn_mat_eigv_2(time_sel,obs_top,edges_centre,edges_linker):
    #time_sel: time point to evaluate
    #obs_top: observed topologies for all time steps
    #edges_centre: get vertices with this number of edges
    #edges_linker:get linkers this number of edges away from centre particles

    
    #all edges at time point time_sel
    edges_ids_t=get_list_of_all_edges(time_sel=time_sel,tops=obs_top)

    #get all vertices with n edges (i.e. all centre particles)
    vertices_with_n_edges_t=vertices_with_n_edges(edges=edges_ids_t,count_edges=edges_centre)

    #get all vertices m edges away from centre particles (i.e. get all linker particles)
    vertices_m_edges_away_t=[]
    for i in range(len(vertices_with_n_edges_t)):
        check_v=vertices_with_n_edges_t[i]
        vertices_m_edges_away_t.append( vertices_n_edges_away(edges=edges_ids_t, start_vertex=check_v,number_edges=edges_linker) )
        
        

    #create adjacency matrix 
    adjacency_matrix_t=create_adjacency_matrix(edges=edges_ids_t, sublists=vertices_m_edges_away_t)
    
    #create connectivity matrix  (i.e. graph laplacian)
    connectivity_matrix_t=np.zeros( (len(adjacency_matrix_t),len(adjacency_matrix_t))  )
    #fill main diagonal with number of edges (i.e. sum of each column)
    for i in range(len(connectivity_matrix_t)):
            connectivity_matrix_t[i][i]=np.sum(adjacency_matrix_t[i])
    #fill off diagonal entries with entries of connectivity matrix    
    for i in range(len(connectivity_matrix_t)):
        for j in range(i): #dont go to main diagonal 
            connectivity_matrix_t[i][j]=adjacency_matrix_t[i][j]*-1
            connectivity_matrix_t[j][i]=adjacency_matrix_t[i][j]*-1


    # Calculate the eigenvalues and eigenvectors
    #eigenvalues, eigenvectors = np.linalg.eig(connectivity_matrix_t)
    components=get_subgraph_eigenvalues(connectivity_matrix_t)

    return connectivity_matrix_t, components

#relaxation time from eigenvalue eigv_lambda
def relax_times(zeta,b,k,T,eigv_lambda):
    tau=(zeta * b**2)/(6 * k * T * eigv_lambda)
    return tau


#storage modulus G' for list of freq omega, from list of N-1 relaxation times tau
def storage_mod(phi,k,T,N,b,tau,omega):
    G_p=[]
    pre_factor=(phi * k * T)/(N * b**3) 
    #get value for each omega
    for i in range(len(omega)):
        G_p_e= np.sum( (omega[i]**2 * tau**2)/(1 + omega[i]**2 * tau**2) )
        G_p.append(G_p_e)
    G_p=np.array(G_p)
    G_p=G_p*pre_factor
    return G_p


#loss modulus G'', dereferenced against solvent viscosity mu_s
#i.e. G'' - omega*mu_s
def loss_mod_deref(phi,k,T,N,b,tau,omega): 
    G_pp_dr=[]
    pre_factor=(phi * k * T)/(N * b**3) 
    #get value for each omega
    for i in range(len(omega)):
        G_pp_dr_e= np.sum( (omega[i] * tau)/(1 + omega[i]**2 * tau**2) )
        G_pp_dr.append(G_pp_dr_e)
    G_pp_dr=np.array(G_pp_dr)
    G_pp_dr=G_pp_dr*pre_factor
    return G_pp_dr


#relaxation modulus G
def relaxation_mod(phi,k,T,N,b,tau,t):
    G=[]
    pre_factor=(phi * k * T)/(N * b**3) 
    #get value for each omega
    for i in range(len(t)):
        G_e= np.sum( np.exp(-t[i]/tau) )
        G.append(G_e)
    G=np.array(G)
    G=G*pre_factor
    return G



#time averaged storage and loss moduli
def get_time_avg_storage_loss_mod(obs_top,time_step,time_skip,omega_t,t,edges_centre,edges_linker,cutoff_zero_eigv,zeta,b,k,T,phi):
    storage_mod_all_t=[]
    loss_mod_all_t=[]
    relaxation_mod_all_t=[]
    conn_mat_all_t=[]
    eigv_all_t=[]
    tau_all_t=[]
    for i in range(len(obs_top[time_skip::time_step])):
        conn_mat_t,eigv_t=get_conn_mat_eigv_1(time_sel=i,obs_top=obs_top[time_skip::time_step],edges_centre=edges_centre,edges_linker=edges_linker)
    
        #get relaxation times for non-zero eigenvalues
        eigv_t_nz=np.array([e for e in eigv_t if abs(e)>cutoff_zero_eigv])
        tau_t=relax_times(zeta=zeta,b=b,k=k,T=T,eigv_lambda=eigv_t_nz)

        #collect all conn. matrices, eigenvalues and relaxation times
        conn_mat_all_t.append(conn_mat_t)
        eigv_all_t.append(eigv_t_nz)
        tau_all_t.append(tau_t)
        
        #get loss, storage and relaxation modulus
        N_non_zero=len(eigv_t_nz)+1
        #N_non_zero=np.count_nonzero(np.any(conn_mat_t, axis=1))
        storage_mod_t=storage_mod(phi=phi,k=k,T=T,N=N_non_zero,b=b,tau=tau_t[:],omega=omega_t)
        loss_mod_t=loss_mod_deref(phi=phi,k=k,T=T,N=N_non_zero,b=b,tau=tau_t[:],omega=omega_t)
        relaxation_mod_t=relaxation_mod(phi=phi,k=k,T=T,N=N_non_zero,b=b,tau=tau_t[:],t=t)
    
        storage_mod_all_t.append(storage_mod_t)
        loss_mod_all_t.append(loss_mod_t)
        relaxation_mod_all_t.append(relaxation_mod_t)

    
    #convert to arrays   
    conn_mat_all_t=np.array(conn_mat_all_t)
    eigv_all_t=np.array(eigv_all_t)
    tau_all_t=np.array(tau_all_t)
    
    storage_mod_all_t=np.array(storage_mod_all_t)
    loss_mod_all_t=np.array(loss_mod_all_t)
    relaxation_mod_all_t=np.array(relaxation_mod_all_t)

    #get mean and stde
    storage_mod_avg_t=np.mean(storage_mod_all_t,axis=0)
    loss_mod_avg_t=np.mean(loss_mod_all_t,axis=0)
    relaxation_mod_avg_t=np.mean(relaxation_mod_all_t,axis=0)
    
    storage_mod_stde_t=np.std(storage_mod_all_t,axis=0)/len(storage_mod_all_t)
    loss_mod_stde_t=np.std(loss_mod_all_t,axis=0)/len(loss_mod_all_t)
    relaxation_mod_stde_t=np.std(relaxation_mod_all_t,axis=0)/len(relaxation_mod_all_t)
    
    #remove imaginary part
    storage_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_avg_t]
    loss_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_avg_t]
    relaxation_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_avg_t]
    
    storage_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_stde_t]
    loss_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_stde_t] 
    relaxation_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_stde_t]
    
    return storage_mod_avg_t_nc,storage_mod_stde_t_nc,loss_mod_avg_t_nc,loss_mod_stde_t_nc,relaxation_mod_avg_t_nc,relaxation_mod_stde_t_nc,conn_mat_all_t,eigv_all_t,tau_all_t
       
    
#time averaged storage and loss moduli
#Version 2: get viscoelastic properties for each sub-component of the graph laplacian
def get_time_avg_storage_loss_mod_2(obs_top,time_step,time_skip,omega_t,t,edges_centre,edges_linker,cutoff_zero_eigv,zeta,b,k,T,phi):
    storage_mod_all_t=[]
    loss_mod_all_t=[]
    relaxation_mod_all_t=[]
    conn_mat_all_t=[]
    eigv_all_t=[]
    tau_all_t=[]
    #print("here",zeta, b,k,T)

    for i in range(len(obs_top[time_skip::time_step])):
        conn_mat_t,components_t=get_conn_mat_eigv_2(time_sel=i,obs_top=obs_top[time_skip::time_step],edges_centre=edges_centre,edges_linker=edges_linker)
    
        #iterate over all components in the graph lapacian at current time step
        
        storage_mod_comp=[]
        loss_mod_comp=[]
        relaxation_comp=[]
        eigv_comp=[]
        tau_comp=[]
        
        #graph laplacian for time step i
        conn_mat_all_t.append(conn_mat_t)
        #print("here",zeta, b,k,T)
        for mm, (size, eigv_t) in enumerate(components_t):
    
            #get relaxation times for non-zero eigenvalues
            
            eigv_t_nz=np.array([e for e in eigv_t if abs(e)>cutoff_zero_eigv])
            if len(eigv_t_nz)>0:
                #print("here",zeta, b,k,T)
                tau_t=relax_times(zeta=zeta,b=b,k=k,T=T,eigv_lambda=eigv_t_nz)
                #tau_t=1/eigv_t_nz

                #collect igenvalues and relaxation times for time step t and component k
                eigv_comp.append(eigv_t_nz)
                tau_comp.append(tau_t)
                #print(tau_t)
                #get loss, storage and relaxation modulus
                N_nodes=len(eigv_t) #number of nodes in graph/objects in topology
                #N_non_zero=len(eigv_t_nz)+1 
                #N_non_zero=np.count_nonzero(np.any(conn_mat_t, axis=1))
                storage_mod_comp_t=storage_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                loss_mod_comp_t=loss_mod_deref(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                relaxation_mod_comp_t=relaxation_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],t=t)
            
                storage_mod_comp.append(storage_mod_comp_t)
                loss_mod_comp.append(loss_mod_comp_t)
                relaxation_comp.append(relaxation_mod_comp_t)
            

        storage_mod_all_t.append(storage_mod_comp)
        loss_mod_all_t.append(loss_mod_comp)
        relaxation_mod_all_t.append(relaxation_comp)
        eigv_all_t.append(eigv_comp)
        tau_all_t.append(tau_comp)
    #convert to arrays   
    #conn_mat_all_t=np.array(conn_mat_all_t)
    #eigv_all_t=np.array(eigv_all_t)
    #tau_all_t=np.array(tau_all_t)
    
    #storage_mod_all_t=np.array(storage_mod_all_t)
    #loss_mod_all_t=np.array(loss_mod_all_t)
    #relaxation_mod_all_t=np.array(relaxation_mod_all_t)

    #get mean and stde
    #storage_mod_avg_t=np.mean(storage_mod_all_t,axis=0)
    #loss_mod_avg_t=np.mean(loss_mod_all_t,axis=0)
    #relaxation_mod_avg_t=np.mean(relaxation_mod_all_t,axis=0)
    
    #storage_mod_stde_t=np.std(storage_mod_all_t,axis=0)/len(storage_mod_all_t)
    #loss_mod_stde_t=np.std(loss_mod_all_t,axis=0)/len(loss_mod_all_t)
    #relaxation_mod_stde_t=np.std(relaxation_mod_all_t,axis=0)/len(relaxation_mod_all_t)
    
    #remove imaginary part
    #storage_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_avg_t]
    #loss_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_avg_t]
    #relaxation_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_avg_t]
    
    #storage_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_stde_t]
    #loss_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_stde_t] 
    #relaxation_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_stde_t]
    
    return storage_mod_all_t,loss_mod_all_t,relaxation_mod_all_t,eigv_all_t,tau_all_t,conn_mat_all_t
    
    
#time averaged storage and loss moduli
#Version 2: get viscoelastic properties for each sub-component of the graph laplacian
#Version 3: normalize relaxation times
def get_time_avg_storage_loss_mod_3(obs_top,time_step,time_skip,omega_t,t,edges_centre,edges_linker,cutoff_zero_eigv,zeta,b,k,T,phi):
    storage_mod_all_t=[]
    loss_mod_all_t=[]
    relaxation_mod_all_t=[]
    conn_mat_all_t=[]
    eigv_all_t=[]
    tau_all_t=[]
    #print("here",zeta, b,k,T)

    for i in range(len(obs_top[time_skip::time_step])):
        conn_mat_t,components_t=get_conn_mat_eigv_2(time_sel=i,obs_top=obs_top[time_skip::time_step],edges_centre=edges_centre,edges_linker=edges_linker)
    
        #iterate over all components in the graph lapacian at current time step
        
        storage_mod_comp=[]
        loss_mod_comp=[]
        relaxation_comp=[]
        eigv_comp=[]
        tau_comp=[]
        
        #graph laplacian for time step i
        conn_mat_all_t.append(conn_mat_t)
        #print("here",zeta, b,k,T)
        for mm, (size, eigv_t) in enumerate(components_t):
    
            #get relaxation times for non-zero eigenvalues
            
            eigv_t_nz=np.array([e for e in eigv_t if abs(e)>cutoff_zero_eigv])
            if len(eigv_t_nz)>0:
                #print("here",zeta, b,k,T)
                tau_t=relax_times(zeta=zeta,b=b,k=k,T=T,eigv_lambda=eigv_t_nz)
                tau_t=tau_t/np.max(tau_t)
                #tau_t=tau_t/np.max(len(eigv_t))
                #collect igenvalues and relaxation times for time step t and component k
                eigv_comp.append(eigv_t_nz)
                tau_comp.append(tau_t)
                #print(tau_t)
                #get loss, storage and relaxation modulus
                N_nodes=len(eigv_t) #number of nodes in graph/objects in topology
                #N_non_zero=len(eigv_t_nz)+1 
                #N_non_zero=np.count_nonzero(np.any(conn_mat_t, axis=1))
                storage_mod_comp_t=storage_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                loss_mod_comp_t=loss_mod_deref(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],omega=omega_t)
                relaxation_mod_comp_t=relaxation_mod(phi=phi,k=k,T=T,N=N_nodes,b=b,tau=tau_t[:],t=t)
            
                storage_mod_comp.append(storage_mod_comp_t)
                loss_mod_comp.append(loss_mod_comp_t)
                relaxation_comp.append(relaxation_mod_comp_t)
            

        storage_mod_all_t.append(storage_mod_comp)
        loss_mod_all_t.append(loss_mod_comp)
        relaxation_mod_all_t.append(relaxation_comp)
        eigv_all_t.append(eigv_comp)
        tau_all_t.append(tau_comp)
    #convert to arrays   
    #conn_mat_all_t=np.array(conn_mat_all_t)
    #eigv_all_t=np.array(eigv_all_t)
    #tau_all_t=np.array(tau_all_t)
    
    #storage_mod_all_t=np.array(storage_mod_all_t)
    #loss_mod_all_t=np.array(loss_mod_all_t)
    #relaxation_mod_all_t=np.array(relaxation_mod_all_t)

    #get mean and stde
    #storage_mod_avg_t=np.mean(storage_mod_all_t,axis=0)
    #loss_mod_avg_t=np.mean(loss_mod_all_t,axis=0)
    #relaxation_mod_avg_t=np.mean(relaxation_mod_all_t,axis=0)
    
    #storage_mod_stde_t=np.std(storage_mod_all_t,axis=0)/len(storage_mod_all_t)
    #loss_mod_stde_t=np.std(loss_mod_all_t,axis=0)/len(loss_mod_all_t)
    #relaxation_mod_stde_t=np.std(relaxation_mod_all_t,axis=0)/len(relaxation_mod_all_t)
    
    #remove imaginary part
    #storage_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_avg_t]
    #loss_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_avg_t]
    #relaxation_mod_avg_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_avg_t]
    
    #storage_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in storage_mod_stde_t]
    #loss_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in loss_mod_stde_t] 
    #relaxation_mod_stde_t_nc=[x.real if isinstance(x, complex) else x for x in relaxation_mod_stde_t]
    
    return storage_mod_all_t,loss_mod_all_t,relaxation_mod_all_t,eigv_all_t,tau_all_t,conn_mat_all_t
#get laplacian matrix
def calculate_graph_laplacian(edges, num_nodes):
    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    
    # Fill the adjacency matrix based on the edges
    for edge in edges:
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    
    # Degree matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    
    # Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix
    
    return laplacian_matrix