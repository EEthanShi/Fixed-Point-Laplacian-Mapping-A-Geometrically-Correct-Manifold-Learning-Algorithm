# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2021 10:47 am

@author: Andi

Performs FLE for manifolds from TC result

Available manifolds: 
    sphere
    parabola
    swiss roll
"""

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import gudhi
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.distance import minkowski
import cvxpy as cp
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
import math

from utils import is_cross, detect_boundary, generate_regular_polygon, sample_spherical, cross
from model import FLE, TC


#%% Setting parameters
data_choice = 'sphere'
N = 500
gamma = 0.1

use_boundary = False
idx_tri = 3

twostepFLE = True # whether we want to use 2-step FLE, this only works when use_boundary = False

save_fig = True
fig_dir = './Experiments/' + data_choice + '/' 

np.random.seed(42)
#%% datasets generate
if data_choice == 'sphere':
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    xi, yi, zi = sample_spherical(N)
    data = np.asarray(np.column_stack((xi,yi,zi)))
    t = data[:,2]
elif data_choice == 'parabola':
    xdata = np.random.random(N)-0.5
    ydata = np.random.random(N)-0.5
    zdata = xdata**2+ydata**2
    data = np.asarray(np.column_stack((xdata,ydata,zdata)))
    t = data[:,2]


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(data[:,0], data[:,1], data[:,2], c = t , cmap = plt.cm.Spectral)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.view_init(30,45)
if save_fig:
    plt.savefig(fig_dir+'{}_3d_pts.pdf'.format(data_choice), bbox_inches='tight')
plt.show()


#%% TC on data
tc = TC(data, 2, True, 0.01, 10)
# construct simp?lex tree
st = tc.create_simplex_tree()
# get triangles (index) and edges (coordinates)
triangles = ([s[0] for s in st.get_skeleton(2) if len(s[0])==3])
edges = []
edge_list=[]
for s in st.get_skeleton(1):
    e = s[0]
    if len(e) == 2:
        edge_list.append([e[0],e[1]])
        edges.append(data[[e[0],e[1]]])

# plot out TC result
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(data[:,0], data[:,1], data[:,2], c = t , cmap = plt.cm.Spectral)
ax.add_collection3d(Line3DCollection(segments=edges, linewidths=0.3))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.view_init(30,45)
if save_fig:
    plt.savefig(fig_dir+ '{}_3d_tri.pdf'.format(data_choice), bbox_inches='tight')
plt.show()

#%% construct graph

G = nx.Graph()
# get nodes 
for i in st.get_skeleton(0):
    v = i[0]
    h = nx.path_graph(v)
    G.add_nodes_from(h)   # vertices
for e in edge_list:
    dist = minkowski(data[e[0]], data[e[1]], p=2) # use minkowski distance, can replace this with other distances 
    G.add_weighted_edges_from( [(e[0], e[1], np.exp(-dist*gamma))] )  #edges 
 
# compute laplacian
adj_mfd = nx.adjacency_matrix(G) 
D = np.sum(adj_mfd, axis = 1)
Lap = np.diag(np.array(D).flatten()) - adj_mfd

#%% Fixed-point LE

if use_boundary:
    # if use boundary from TC as fixed points
    boundary_edge, boundary_point_idx = detect_boundary(st, edge_list)
    
    # if there is no boundary, we raise an error
    assert len(boundary_point_idx) > 0, "No boundary detected, please do not use boundary"
    
    boundary_point = data[boundary_point_idx] # coordinates of boundary points
    
    # Highlight the boundary points
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot points
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c = t , cmap = plt.cm.Spectral)
    ax.scatter3D(boundary_point[:,0], boundary_point[:,1], boundary_point[:,2], c = 'black', marker = '*') # boundary points
    ax.add_collection3d(Line3DCollection(segments=data[boundary_edge, :], linewidths=0.3)) # boundary edges
    ax.view_init(60,0)
    plt.show()
    
    
    # ===== Map boundary in cyclic order to boundary of regular polygon =======
    # Regular polygon as fixed points 
    C = generate_regular_polygon(len(boundary_point_idx))
    # Note we need to match the index in cyclic order, we first start with any edge
    e_temp = boundary_edge[0]
    C_index = [e_temp[0], e_temp[1]]
    
    # start with iteration 2 because we already add 2 vertices
    iteration = 2
    while iteration <= len(boundary_point_idx):
        last_pt = C_index[-1]
        # iterate thourgh all edges 
        for e in boundary_edge:
            if last_pt in e:
                pt_candidate =  set(e).difference(set(C_index)) # set difference
                if len(pt_candidate) == 1:
                    # this means we have a new point to add
                    C_index.append(list(pt_candidate)[0])
        iteration += 1     
    # ========================================================================
    
else:
    # map any triangle into a [[1,0], [-1,0], [0,1]] fixed point
    C = np.matrix([[1,0],[-1,0],[0,1]])
    C_index = triangles[idx_tri]
    

Y = FLE(Lap, C, C_index)


#%% Plots
fle_edges=[]
for s in st.get_skeleton(1):
    e = s[0]
    if len(e) == 2:
        fle_edges.append(Y[[e[0],e[1]]])
        
plt.scatter(Y[:,0], Y[:,1],c =t, cmap = plt.cm.Spectral)
lc = LineCollection(segments = fle_edges, linewidths=0.3)
plt.gca().add_collection(lc)
plt.xticks([], [])
plt.yticks([], [])
if save_fig:
    plt.savefig(fig_dir + data_choice + '_FLE_1step.pdf', bbox_inches='tight')
plt.show()    

#%%
crosses = cross(fle_edges, edge_list, findall=True)
if len(crosses) != 0:
    print('{} cross found!'.format(len(crosses)))
else:
    print('No cross found!')


#%% Second-step Fixed-point LE

if twostepFLE and (not use_boundary):
    
    # detect boundary again because we haven't done so if use_boundary == False
    boundary_edge, boundary_point_idx = detect_boundary(st, edge_list)
    # if there is no boundary, we raise an error (note error msg different)
    assert len(boundary_point_idx) > 0, "No boundary detected, one-step FLE is sufficient"
    
    # plot out the boundary in the first-step fle
    plt.scatter(Y[:,0], Y[:,1], c = t , cmap = plt.cm.Spectral)
    lc = LineCollection(segments = fle_edges, linewidths=0.3)
    plt.gca().add_collection(lc)
    plt.scatter(Y[boundary_point_idx,0], Y[boundary_point_idx,1], c='black', marker='*')
    plt.xticks([], [])
    plt.yticks([], [])
    if save_fig:
        plt.savefig(fig_dir + data_choice + '_FLE_1step_bd.pdf', bbox_inches='tight')
    plt.show()  
    
    # plot the boundary in high dimensional space
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c = t, cmap = plt.cm.Spectral)
    ax.add_collection3d(Line3DCollection(segments = data[edge_list], linewidths=0.3 ))
    ax.scatter3D(data[boundary_point_idx,0], data[boundary_point_idx,1], data[boundary_point_idx,2], c='black', marker='*')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(30,45)
    if save_fig:
        plt.savefig(fig_dir + data_choice + '_FLE_3d_bd.pdf', bbox_inches='tight')
    plt.show()
    
    
    # Fixed point updated
    C = np.matrix([Y[i] for i in boundary_point_idx])
    C_index = np.array(list(boundary_point_idx)) # index for fixed points
    
    Y = FLE(Lap, C, C_index)
    
    # Final Plot
    fle_edges=[]
    for e in edge_list:    
        fle_edges.append(Y[[e[0],e[1]]])
            
    plt.scatter(Y[:,0], Y[:,1],c = t , cmap = plt.cm.Spectral)
    lc = LineCollection(segments = fle_edges, linewidths=0.3)
    plt.gca().add_collection(lc)
    plt.xticks([], [])
    plt.yticks([], [])
    if save_fig:
        plt.savefig(fig_dir + data_choice + '_FLE_2step.pdf', bbox_inches='tight')
    plt.show()  
    
    # Final Check crossing
    crosses = cross(fle_edges, edge_list, findall=True)
    if len(crosses) != 0:
        print('{} cross found!'.format(len(crosses)))
    else:
        print('No cross found!')


