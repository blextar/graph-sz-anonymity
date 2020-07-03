# Source: https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions/blob/master/graph_reducer/
# Note: target_k was added wrt the original repository to ensure that the partition is refined until the required k is reached

import szemeredi_regularity_lemma as srl
import partition_initialization
import conditions
import refinement_step as rs

def generate_szemeredi_reg_lemma_implementation(kind, sim_mat, epsilon, is_weighted, random_initialization,
                                                refinement, drop_edges_between_irregular_pairs, target_k):
    """
    generate an implementation of the Szemeredi regularity lemma for the graph summarization
    :param kind: the kind of implementation to generate. The currently accepted strings are 'alon' for the Alon
                 the Alon implementation, and 'frieze_kannan' for the Frieze and Kannan implementation
    :param sim_mat: the similarity matrix representing the graph
    :param epsilon: the epsilon parameter to determine the regularity of the partition
    :param is_weighted: set it to True to specify if the graph is weighted
    :param random_initialization: set it to True to generate a random partition of the graph
    :param random_refinement: set it to True to randomly re-asset the nodes in the refinement step
    :param is_fully_connected_reduced_matrix: if set to True the similarity matrix is not thresholded and a fully
           connected graph is generated
    :param is_no_condition_considered_regular: if set to True, when no condition
    :param target_k: the cardinality of the target partition
    :return:
    """
    alg = srl.SzemerediRegularityLemma(sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs,target_k)

    if refinement == 'indeg_guided':
        alg.refinement_step = rs.indeg_guided
    if refinement == 'degree_based':
        alg.refinement_step = rs.degree_based

    if random_initialization:
        alg.partition_initialization = partition_initialization.random
    else:
        alg.partition_initialization = partition_initialization.degree_based

    if kind == "alon":
        alg.conditions = [conditions.alon1, conditions.alon3, conditions.alon2]
        #alg.conditions = [conditions.alon1, conditions.alon3]
    elif kind == "frieze_kannan":
        alg.conditions = [conditions.frieze_kannan]
    else:
        raise ValueError("Could not find the specified graph summarization method")

    return alg
