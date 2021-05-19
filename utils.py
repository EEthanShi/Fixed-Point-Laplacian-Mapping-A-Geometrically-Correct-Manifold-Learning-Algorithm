# -*- coding: utf-8 -*-
"""
Created on 27/03/2021 9:35 am

@author: Andi
"""

import numpy as np
import math

#%% Detect crossing
class Point: 
	def __init__(self, x, y): 
		self.x = x 
		self.y = y 

# Given three colinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r): 
	if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
		(q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
		return True
	return False

def orientation(p, q, r): 
	# to find the orientation of an ordered triplet (p,q,r) 
	# function returns the following values: 
	# 0 : Colinear points 
	# 1 : Clockwise points 
	# 2 : Counterclockwise 

	
	val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
	if (val > 0): 
		
		# Clockwise orientation 
		return 1
	elif (val < 0): 
		
		# Counterclockwise orientation 
		return 2
	else: 
		
		# Colinear orientation 
		return 0

# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
	
	# Find the 4 orientations required for 
	# the general and special cases 
	o1 = orientation(p1, q1, p2) 
	o2 = orientation(p1, q1, q2) 
	o3 = orientation(p2, q2, p1) 
	o4 = orientation(p2, q2, q1) 

	# General case 
	if ((o1 != o2) and (o3 != o4)): 
		return True
	if ((p1.x==p2.x) and (p1.y==p2.y)):
		return False
	if ((p2.x ==q2.x) and (p2.y ==q2.y)):
		return False 

	# Special Cases 

	# p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    
	if ((o1 == 0) and onSegment(p1, p2, q1)): 
		return False

	# p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
	if ((o2 == 0) and onSegment(p1, q2, q1)): 
		return False

	# p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
	if ((o3 == 0) and onSegment(p2, p1, q2)): 
		return False

	# p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
	if ((o4 == 0) and onSegment(p2, q1, q2)): 
		return False

    
	# If none of the cases 
	return False


def is_cross(fle_edges):
    line_cross=[]
    intersect_lines=[]
    for i,j in enumerate(fle_edges):
        testing_edge = fle_edges[i]
        targeting_edge = fle_edges[i+1:] # remove the duplications of checking
        a = Point(testing_edge[0][0],testing_edge[0][1])
        b =Point(testing_edge[1][0],testing_edge[1][1])     
       
        for j in targeting_edge:
            c =Point(j[0][0],j[0][1])
            d =Point(j[1][0],j[1][1])
            line_cross.append(doIntersect(a,b,c,d))
            if doIntersect(a,b,c,d)>0:
                intersect_lines.append([[a.x,a.y],[b.x,b.y],[c.x,c.y],[d.x,d.y]]) # crossed lines with duplications and colinear points 
    
    remove=[]
    duplicated_cross = []
    for i in intersect_lines:
       if len(np.unique(i)) != np.array(i).shape[0]*np.array(i).shape[1]:
           remove.append(i)
    duplicated_cross = [j for j in intersect_lines if j not in remove] # remove crosses from two edges with same vertices 
    
    final_cross = []
    c=[]
    for i in duplicated_cross:
        c.append(set(np.unique(i)))
    for i in c:
        if i not in final_cross:
            final_cross.append(list(i)) # remove duplicated cross records; this is double of the cross value.  
    line = []
    for i in list(final_cross):
        line.append(np.array(i).reshape(4,2))
    if len(final_cross) !=0: 
        #print('line cross detected and the number of pairs is'+str(len(final_cross)/4))
        return len(final_cross)/4
    
    if len(final_cross)/4==0:
        #print ('good, no cross detected')
        return False
    
#%% detect_crossing_andy

# Used for detecting crossing in 2D space
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
def intersect(A,B,C,D):
    # Return true if line segments AB and CD intersect
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def cross(edges, indexes, findall = True):
    """
        Input: 
            edges: a list of edges containing coordinates of each endpoint
            indexes: a list of edges containing indexes of each endpoint
            findall: if True, find all crossings, if false, find the first crossing and break
        Output:
            A list of crossings             
    """
    n_edges = len(edges)
    
    out = []
    for idx_1 in range(n_edges):
        
        PointIndex_a = indexes[idx_1][0]
        PointIndex_b = indexes[idx_1][1]
        for idx_2 in range(idx_1+1, n_edges):
            if (PointIndex_a not in indexes[idx_2]) and (PointIndex_b not in indexes[idx_2]):
                # excluding the case where there is a common end point
                line1 = edges[idx_1]
                line2 = edges[idx_2]
                if intersect(line1[0], line1[1], line2[0], line2[1]):
                    out.append([line1, line2])
                    
                    if not findall:
                        return out  
    return out


# Used for detecting crossings in 3D
def intersect3D(A,B,C,D):
    # reference: https://stackoverflow.com/questions/55220355/how-to-detect-whether-two-segmentin-3d-spaceintersect
    a1 = A[0]; a2 = A[1]; a3 = A[2]; b1 = B[0]; b2 = B[1]; b3 = B[2];
    c1 = C[0]; c2 = C[1]; c3 = C[2]; d1 = D[0]; d2 = D[1]; d3 = D[2];
    
    a=b1-a1
    b=c1-d1
    c=c1-a1
    d=b2-a2
    e=c2-d2
    f=c2-a2

    #find t and s using formula
    if ((e*a-b*d) < 1e-6) or ((d*b-a*e) < 1e-6):
        return False
    else:
        t=(c*e-f*b)/(e*a-b*d)
        s=(d*c-a*f)/(d*b-a*e)

    #check if third equation is also satisfied(we have 3 equations and 2 variable
    return ((t*(b3-a3)+s*(c3-d3))==c3-a3) and (0<= t <= 1 and 0 <= s <= 1)


def cross3D(edges, indexes, findall = True):
    """
        Input: 
            edges: a list of edges containing coordinates of each endpoint
            indexes: a list of edges containing indexes of each endpoint
            findall: if True, find all crossings, if false, find the first crossing and break
        Output:
            A list of crossings             
    """
    n_edges = len(edges)
    
    out = []
    for idx_1 in range(n_edges):
        
        PointIndex_a = indexes[idx_1][0]
        PointIndex_b = indexes[idx_1][1]
        for idx_2 in range(idx_1+1, n_edges):
            if (PointIndex_a not in indexes[idx_2]) and (PointIndex_b not in indexes[idx_2]):
                # excluding the case where there is a common end point
                line1 = edges[idx_1]
                line2 = edges[idx_2]
                if intersect3D(line1[0], line1[1], line2[0], line2[1]):
                    print(intersect3D(line1[0], line1[1], line2[0], line2[1]))
                    out.append([line1, line2])
                    
                    if not findall:
                        return out  
    return out
        

#%%
def detect_boundary(st, edge_list):
    """
    Finding the boundary of TC result
        Input:
            st: a tc simplex_tree object on the data
            edge_list: all edges where each vertex is an index
        Output:
            boundary_edge: all edges on the boundary
            boundary_point_idx: all point index that are on the boundary
    """
    # finding the boundary edge
    boundary_edge = []
    boundary_point_idx = []
    for e in edge_list:
        edge_star_length = len(st.get_star(e))-1
        if edge_star_length == 1:
            boundary_edge.append(e)
            boundary_point_idx = boundary_point_idx + e # append list to a list
    
    boundary_point_idx = np.unique(boundary_point_idx)
    
    return boundary_edge, boundary_point_idx

def detect_boundary3D(st, edge_list):
    boundary_edge = []
    boundary_point_idx = []
    for e in edge_list:
        star = [s[0] for s in st.get_star(e) if len(s[0]) == 3]
        if len(star) == 2:
            boundary_edge.append(e)
            boundary_point_idx = boundary_point_idx + e # append list to a list
    
    boundary_point_idx = np.unique(boundary_point_idx)
    
    return boundary_edge, boundary_point_idx
    

#%%

def generate_regular_polygon(numSides, rotation = 0.0, radius=1.0, x =1.0, y = 1.0):
    """
    Generate a regular polygon with center as the origin.
    References: https://nerdparadise.com/programming/pygameregularpolygon
        Input:
            numSides: number of sides/vertices
            rotation: rotation angle, default = 0.0
            radius: radius, default = 1.0
            x: starting x-axis, default = 1.0
            y: starting y-axis, default = 1.0
        Output:
            A set of points as vertices of the regular polygon
    """
    vertices = []
    for i in range(numSides):
        x = x + radius * math.cos(rotation + math.pi * 2 * i / numSides)
        y = y + radius * math.sin(rotation + math.pi * 2 * i / numSides)
        vertices.append([x, y])
    return np.array(vertices)
        

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec