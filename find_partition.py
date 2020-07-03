# Authors: Luca Rossi, Giorgia Minello, and Daniele Foffano
# Reference: k-Anonymity on Graphs using the Szemere ́di Regularity Lemma (under review)
# Note: the directory regularity_lemma contains the a slightly modified version of the code available at
# https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions/blob/master/graph_reducer/
# Modifications (in szemeredi_lemma_builder.py, szemeredi_regularity_lemma.py, and partition_initialization.py) are marked in the code

import sys
if './regularity_lemma/' not in sys.path:
    sys.path.insert(1, './regularity_lemma/')

import szemeredi_lemma_builder as sz
import numpy as np
'''
    Function to find a Szemeredi regular partition with a cardinality equal to the k inserted by the user.
    The k must be a power of two.
    Multiple partitions are examined, using different epsilons within a range going from 0.25 to 0.01.
    The chosen partition will be the best partition found for that k, comparing the epsilon of each partition found (the smallest, the best).

    Function parameters:
        * target_k : the k inserted by the user
        * adj_matrix : the adjacency matrix of the graph
    
    Function output: 
        * a dictionary with the following structure:
            {
                'k' : the number of partitions
                'epsilon' : the epsilon of the partition
                'classes' : array of size n (number of nodes in the graph); the node i belongs to the class classes[i] 
                'sze_idx' : the szemeredy_index used to evaluate the quality of the partition
                'irr_pairs' : number of irregular pairs
                'irr_list' : a list of the irregular pairs
            }
'''

def find_partition(target_k, adj_matrix, heuristic, min_epsilon=0.01, max_epsilon=0.26, step=0.002, verbose=False):

    if( target_k == 0 or not(target_k & (target_k-1) == 0)): # Check if k is not power of 2
        print("The cardinality of the partition you seek should be a power of 2")
        return {}
    else:
        epsilon_range = np.arange(start=min_epsilon, stop=max_epsilon, step=step)
        best_partition = {}
        iterations = 10
        for iteration in range(iterations):
            for epsilon in epsilon_range:
                alg = sz.generate_szemeredi_reg_lemma_implementation('alon', adj_matrix, epsilon, False, True, heuristic, False, target_k)
                is_regular, k, classes, sze_idx, regularity_list, irr_pairs, irr_list = alg.run(verbose=verbose)
                
                if is_regular:
                    print("k=" + str(k) + " epsilon=" + str(epsilon) + " sze_idx=" + str(sze_idx) +  " irr_pairs=" + str(irr_pairs))
                    if best_partition == {}:
                        best_partition = {'k' : k,'epsilon' : epsilon,'classes' : classes,'sze_idx' : sze_idx,'irr_pairs' : irr_pairs,'irr_list' : irr_list}
                        print("*** best partition set ***")
                    #elif irr_pairs < best_partition['irr_pairs']:
                    elif epsilon < best_partition['epsilon'] or (epsilon == best_partition['epsilon'] and irr_pairs < best_partition['irr_pairs']):
                        best_partition = {'k' : k,'epsilon' : epsilon,'classes' : classes,'sze_idx' : sze_idx,'irr_pairs' : irr_pairs,'irr_list' : irr_list}
                        print("*** best partition updated ***")
                    break
        if best_partition == {}:
            print("I couldn't find a partition for k=" + str(target_k))
        return best_partition
