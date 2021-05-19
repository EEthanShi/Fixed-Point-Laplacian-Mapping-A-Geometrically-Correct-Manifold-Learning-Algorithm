# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:59:30 2021

@author: Andi Han
"""

import networkx as nx
import numpy as np
import gudhi
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.distance import minkowski
import cvxpy as cp
from matplotlib.collections import LineCollection
import math
import scipy.io
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from itertools import combinations
import gudhi

from utils import is_cross, detect_boundary, sample_spherical, cross
from model import FLE, TC, myIsomap, myLocallyLinearEmbedding, Autoencoder


from sklearn.manifold import SpectralEmbedding, MDS, TSNE

import torch
from torch import nn

import glob, os


#%%

gamma = 0.1

data_choice = 'monkey_saddle'


data_TC = ['sphere', 'parabola']
data_DT = ['swiss_roll', 'twinpeaks', 'monkey_saddle']

save_fig = True
fig_dir = './Experiments/' + data_choice + '/other_methods/' 

N = 500
d = 2
D = 3

np.random.seed(42)
torch.manual_seed(42)
#%% load datasets
if data_choice == 'twinpeaks':
    
    inc = 1.5 / np.sqrt(N);
    data_2d = 1 - 2 * np.random.rand(N, 2);
    data = np.asarray(np.column_stack((data_2d[:,0],data_2d[:,1], np.sin(np.pi * data_2d[:,0]) * np.tanh(3 * data_2d[:,1]))))
    data[:,2] = data[:,2] * 10
    t = data_2d[:,0]
    
elif data_choice == 'swiss_roll':
    phi = 1.5 * np.pi * (1 + 2 * np.random.random(N)) 
    theta = 30* np.random.random(N)
    xdata = np.ravel(phi * np.cos(phi))
    ydata = np.ravel(theta)
    zdata = np.ravel(phi * np.sin(phi))
    data = np.asarray(np.column_stack((xdata,ydata,zdata)))
    data_2d = np.asarray(np.column_stack((np.ravel(phi), np.ravel(theta))))
    t = np.ravel(phi) # let t to represent unique identifier for ploting

elif data_choice == 'monkey_saddle':
    xdata = np.random.random(N)-0.5
    ydata = np.random.random(N)-0.5
    zdata =xdata**3-3*xdata*ydata**2
    data = np.asarray(np.column_stack((xdata,ydata,zdata)))
    data_2d = data[:,:2]
    t = data_2d[:,0]
    
elif data_choice == 'sphere':
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

#%% generate triangulation
if data_choice in data_TC:
    
    tc = TC(data, 2, True, 0.01, 10)
    # construct simp?lex tree
    st = tc.create_simplex_tree()
    # get triangles (index) and edges (coordinates)
    triangles = ([s[0] for s in st.get_skeleton(2) if len(s[0])==3])
    edge_list=[]
    for s in st.get_skeleton(1):
        e = s[0]
        if len(e) == 2:
            edge_list.append([e[0],e[1]])
            
elif data_choice in data_DT:
    
    dt = gudhi.AlphaComplex(points = data_2d)

    # create simplex tree
    st = dt.create_simplex_tree()
    print('Triangulation finished')
    
    # triangles, edges
    triangles = ([s[0] for s in st.get_skeleton(2) if len(s[0])==3])
    edge_list=[]
    for s in st.get_skeleton(1):
        e = s[0]
        if len(e) == 2:
            edge_list.append([e[0],e[1]])

else:
    raise Exception('Please specify a valid data choice')


# 3d tri plots
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(data[:,0], data[:,1], data[:,2], c = t , cmap = plt.cm.Spectral)
ax.add_collection3d(Line3DCollection(segments = data[edge_list], linewidths=0.3))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.view_init(30,45)
plt.show()


#%% construct graph

G = nx.Graph()
G_dist = nx.Graph()
# get nodes 
for i in st.get_skeleton(0):
    v = i[0]
    h = nx.path_graph(v)
    G.add_nodes_from(h)   # vertices
    G_dist.add_nodes_from(h)
for e in edge_list:
    dist = minkowski(data[e[0]], data[e[1]], p=2) # use minkowski distance, can replace this with other distances 
    G.add_weighted_edges_from( [(e[0], e[1], np.exp(-dist*gamma))] )  #edges 
    G_dist.add_weighted_edges_from([(e[0], e[1], dist)] )
 
# compute laplacian
adj_mfd = nx.adjacency_matrix(G) 
dist_mfd = nx.adjacency_matrix(G_dist) 



#%%

methods = ['LE', 'ISOMAP', 'LLE', 'LTSA', 'MDS', 'TSNE', 'AE']


for method in methods:
    
    if method == 'LE':
        data_transformed = SpectralEmbedding(n_components=d, affinity='precomputed', random_state=0).fit_transform(adj_mfd)
        
    elif method == 'MDS':
        data_transformed = MDS(n_components=d).fit_transform(data)
        
    elif method == 'TSNE':
        data_transformed = TSNE(n_components=d, init='pca', random_state=0).fit_transform(data)
        
    elif method == 'ISOMAP':
        data_transformed = myIsomap(n_components=d).fit_transform(dist_mfd)
        
    elif method == 'LLE':
        data_transformed = myLocallyLinearEmbedding(n_components=d, eigen_solver= 'auto').fit_transform(adj_mfd,data)
        
    elif method =='LTSA':
        data_transformed = myLocallyLinearEmbedding(n_components=d, eigen_solver= 'auto', method='ltsa').fit_transform(adj_mfd,data)
        
    elif method == 'AE':        
        
        data_torch = torch.from_numpy(data).float()
        
        AE = Autoencoder(D,d)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(AE.parameters(), lr = 0.1)
        
        epochs = 1000
        for i in range(epochs):
            optimizer.zero_grad()
            
            output = AE(data_torch)
            loss = loss_fn(output, data_torch)
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print('Epoch {}: {:.4f}'.format(i, loss))
        
        AE.eval()
        with torch.no_grad():
            data_transformed = AE.encode(data_torch).detach().numpy()
                          
    else:
        raise Exception('Please specify a valid solver!')
            
    
    edges2d=[]
    for e in edge_list:    
        edges2d.append(data_transformed[[e[0],e[1]]])
        
    # check crosses
    crosses = cross(edges2d, edge_list, findall=True)
    print('{}: {} crosses'.format(method, len(crosses)))
    
    plt.scatter(data_transformed[:,0], data_transformed[:,1],c = t , cmap = plt.cm.Spectral)
    lc = LineCollection(segments = edges2d, linewidths=0.3)
    plt.gca().add_collection(lc)
    plt.xticks([], [])
    plt.yticks([], [])
    if save_fig:
        
        filels = glob.glob(fig_dir + '*{}*.pdf'.format(method))
        if len(filels) != 0:
            for file in filels:
                os.remove(file)
        
        plt.savefig(fig_dir + data_choice + '_{}_{}.pdf'.format(method, len(crosses)), bbox_inches='tight')
    plt.show()  





