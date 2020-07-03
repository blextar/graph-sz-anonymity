# Authors: Luca Rossi, Giorgia Minello, and Daniele Foffano
# Reference: k-Anonymity on Graphs using the Szemere ́di Regularity Lemma (under review)

import numpy as np
import random as rdm
from collections import defaultdict
from find_partition import find_partition

import networkx as nx

def density(subset, adj_matrix, oriented):

    n = len(subset)
    inside_edges = 0
    possible_edges = (n*(n-1))
    edges = []
    if n == 1:
        return 0, []
    
    if not oriented:
        while(len(subset) != 0):
            node1 = subset.pop()
            for node2 in subset:
                edges.append((node1, node2))
                if(adj_matrix[node1][node2] == 1):
                    inside_edges += 2
    else:
        while(len(subset) != 0):
            node1 = subset.pop()
            for node2 in subset:
                
                edges.append((node1, node2))
                edges.append((node2, node1))
                
                if(adj_matrix[node1][node2] == 1):
                    inside_edges += 1
                    
                if(adj_matrix[node2][node1] == 1):
                    inside_edges += 1
        
    return inside_edges/possible_edges, edges

'''
    Function using the Erdos-Renyi model to randomize a set of edges of a group of nodes.
    The probability used is the inner edges density before the randomization.

    Function parameters:
        * p : the probability used by the Erdos-Renyi model
        * edges : the set of edges to randomize
        * adj_matrix : the adjacency matrix of the graph
        * oriented : boolean value. True if the graph is oriented, false otherwise.

    Function output:
        * adj_matrix : the adjacency matrix of the graph, with the randomized edges
'''

def Erdos_Renyi(p, edges, adj_matrix, oriented):
    
    if not oriented:
        for (n1, n2) in edges:
            i = rdm.random()
            if(i <= p):
                adj_matrix[n1][n2] = 1
                adj_matrix[n2][n1] = 1
            else:
                adj_matrix[n1][n2] = 0
                adj_matrix[n2][n1] = 0
    else:
        for (n1, n2) in edges:
            i = rdm.random()
            if(i <= p):
                adj_matrix[n1][n2] = 1
            else:
                adj_matrix[n1][n2] = 0
    return adj_matrix

'''
    Function using the Erdos-Renyi model to randomize the edges going from one group of nodes to another
    (i.e.: that have one vertex in the first group and the other in the second one).
    The probability used by the model is the density of the edges between these two groups.

    Function parameters:
        * group1 : the first group of nodes
        * group2 : the second group of nodes
        * adj_matrix : the adjacency matrix of the graph
        * oriented : boolean value. True if the graph is oriented, false otherwise.

    Function output:
        * adj_matrix : the adjacency matrix of the graph, with the randomized edges
'''

def anonymize_irr_couple(group1, group2, adj_matrix, oriented):
    
    possible_edges = len(group1) * len(group2) * 2
    between_edges = 0
    
    if not oriented:
        for n1 in group1:
            for n2 in group2:
                if(adj_matrix[n1][n2] == 1):
                    between_edges += 2
        density = between_edges/possible_edges
        
        for n1 in group1:
            for n2 in group2:
                i = rdm.random()
                if(i <= density):
                    adj_matrix[n1][n2] = 1
                    adj_matrix[n2][n1] = 1
                else:
                    adj_matrix[n1][n2] = 0
                    adj_matrix[n2][n1] = 0
                    
    else:
        for n1 in group1:
            for n2 in group2:
                if(adj_matrix[n1][n2] == 1):
                    between_edges += 1 
                    
                if(adj_matrix[n2][n1] == 1):
                    between_edges += 1
                    
        density = between_edges/possible_edges
        
        for n1 in group1:
            for n2 in group2:
                i = rdm.random()
                if(i <= density):
                    adj_matrix[n1][n2] = 1
                else:
                    adj_matrix[n1][n2] = 0
                    
                i = rdm.random()
            
                if(i <= density):
                    adj_matrix[n2][n1] = 1
                else:
                    adj_matrix[n2][n1] = 0
                    
    return adj_matrix

'''
    Function to anonymize a graph using the Szemeredi regularity lemma and the Erdos-Renyi model.

    Function parameters:

        * adj_matrix: the adjacency matrix of the graph
        * k : the number of partitions desired by the user (only powers of two are accepted)
        * oriented : boolean value. True if the graph is oriented, false otherwise.

    Function output:

        * adj_matrix : the adjacency matrix of the randomized graph
'''

def anonymize(adj_matrix, k, oriented, heuristic, verbose=False):   
    
    ''' Partitioning the graph using the Szemeredi regularity lemma'''
    
    sz_partition = find_partition(k, adj_matrix, heuristic, verbose=verbose)
    
    if sz_partition == {}:
        print("Could not find any partition of the given cardinality.")
        return None
    else:
        # Grouping nodes by partition
        nodes_class = defaultdict(list)
        
        for i in range(len(sz_partition['classes'])):
            nodes_class[sz_partition['classes'][i]].append(i)

        # Anonymizing each partition inner edges, using Erdos-Renyi model.
        partitions_density = {}
        partitions_edges = {}
        
        for (c, nodes) in nodes_class.items():
            partitions_density[c], partitions_edges[c] = density(nodes.copy(), adj_matrix, oriented)
            
        for (c, edges) in partitions_edges.items():
            adj_matrix = Erdos_Renyi(partitions_density[c], edges, adj_matrix, oriented)
        
        # Anonymizing irregular partitions outer edges, using Erdos-Renyi model.
        for (g1, g2) in sz_partition['irr_list']:
            adj_matrix = anonymize_irr_couple(nodes_class[g1], nodes_class[g2], adj_matrix, oriented)
            
        return adj_matrix, sz_partition['epsilon'], sz_partition['sze_idx'], sz_partition['irr_pairs'], sz_partition['irr_list'], sz_partition['classes']