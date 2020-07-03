# Source: https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions/blob/master/graph_reducer/
# Note:
# - target_k was added wrt the original repository to ensure that the partition is refined until the required k is reached
# - irr_list has been added wrt the original repository to return a list of the irregular pairs that need to be anonymized
# - the method "run" has been modified wrt to the original repository as described below
import random
import numpy as np

from classes_pair import ClassesPair
from classes_pair import WeightedClassesPair
#import ipdb


class SzemerediRegularityLemma:
    # Methods required by the algorithm
    partition_initialization = None
    refinement_step = None
    conditions = []

    # Main data structure
    sim_mat = np.empty((0, 0), dtype='float32')
    adj_mat = np.empty((0, 0), dtype='float32')
    reduced_sim_mat = np.empty((0, 0), dtype='float32')
    classes = np.empty((0, 0), dtype='int16')
    degrees = np.empty((0, 0), dtype='int16')
    
    # Parameters of the algorithm
    N = 0
    k = 0
    classes_cardinality = 0
    epsilon = 0.0
    sze_idx = 0.0
    target_k = 0

    # Auxiliary structures used to keep track of the relation between the classes of the partition
    certs_compls_list = []
    """structure containing the certificates and complements for each pair in the partition"""
    regularity_list = []
    """list of lists of size k, each element i contains the list of classes regular with class i"""
    irr_list = []
    """list of containing irregular pairs"""
    
    # Flags
    is_weighted = False
    drop_edges_between_irregular_pairs = False

    def __init__(self, sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs,target_k):
        if is_weighted:
            self.sim_mat = sim_mat
            self.adj_mat = (sim_mat > 0).astype("int8")
        else:
            self.adj_mat = sim_mat

        self.epsilon = epsilon
        self.N = self.adj_mat.shape[0]
        self.degrees = np.argsort(self.adj_mat.sum(0))
        self.target_k = target_k

        # Flags
        self.is_weighted = is_weighted
        self.drop_edges_between_irregular_pairs = drop_edges_between_irregular_pairs

    def generate_reduced_sim_mat(self):
        """
        generate the similarity matrix of the current classes
        :return sim_mat: the reduced similarity matrix
        """
        self.reduced_sim_mat = np.zeros((self.k, self.k), dtype='float32')

        for r in range(2, self.k + 1):
            for s in (range(1, r) if not self.drop_edges_between_irregular_pairs else self.regularity_list[r - 2]):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)
                self.reduced_sim_mat[r - 1, s - 1] = self.reduced_sim_mat[s - 1, r - 1] = cl_pair.bip_density


    def check_pairs_regularity(self):
        #print("hi 3")
        """
        perform step 2 of Alon algorithm, determining the regular/irregular pairs and their certificates and complements
        :return certs_compls_list: a list of lists containing the certificate and complement
                for each pair of classes r and s (s < r). If a pair is epsilon-regular
                the corresponding complement and certificate in the structure will be set to the empty lists
        :return num_of_irregular_pairs: the number of irregular pairs
        """

        num_of_irregular_pairs = 0
        self.sze_idx = 0.0
        self.irr_list = []
        #print("hello")
        for r in range(2, self.k + 1):
            self.certs_compls_list.append([])
            self.regularity_list.append([])
            
            
            for s in range(1, r):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                is_verified = False
                for i, cond in enumerate(self.conditions):
                    #print("FACCIO ITERAZIONE")
                    is_verified, cert_pair, compl_pair = cond(self, cl_pair)
                    if is_verified:
                        self.certs_compls_list[r - 2].append([cert_pair, compl_pair])

                        if cert_pair[0]:
                            #print("if")
                            #print(r,s)
                            self.irr_list.append((r,s))
                            num_of_irregular_pairs += 1
                        else:
                            #print("else")
                            self.regularity_list[r - 2].append(s)

                        break

                if not is_verified:
                    # if no condition was verified then consider the pair to be regular
                    self.certs_compls_list[r - 2].append([[[], []], [[], []]])

                self.sze_idx += cl_pair.bip_density ** 2.0

        self.sze_idx *= (1.0 / self.k ** 2.0)
        return num_of_irregular_pairs

    def check_partition_regularity(self, num_of_irregular_pairs):
        """
        perform step 3 of Alon algorithm, checking the regularity of the partition
        :param num_of_irregular_pairs: the number of found irregular pairs in the previous step
        :return: True if the partition is regular, False otherwise
        """
        #print("hi 2")
        return num_of_irregular_pairs <= self.epsilon * ((self.k * (self.k - 1)) / 2.0)
    
    # Note: this method has been modfied wrt to the original repository to allow the refinement step to be applied to
    # irregular as well as regular partition (as opposed to irregular partitions only in the code of Fiorucci et al.)
    def run(self, b=2, compression_rate=0.5, iteration_by_iteration=False, verbose=False):
        """
        run the Alon algorithm.
        :param b: the cardinality of the initial partition (C0 excluded)
        :param compression_rate: the minimum compression rate granted by the algorithm, if set to a value in (0.0, 1.0]
                                 the algorithm will stop when k > int(compression_rate * |V|). If set to a value > 1.0
                                 the algorithm will stop when k > int(compression_rate)
        :param iteration_by_iteration: if set to true, the algorithm will wait for a user input to proceed to the next
               one
        :param verbose: if set to True some debug info is printed
        :return the reduced similarity matrix
        """
        if 0.0 < compression_rate <= 1.0:
            max_k = int(compression_rate * self.N)
        elif compression_rate > 1.0:
            max_k = int(compression_rate)
        else:
            raise ValueError("incorrect compression rate. Only float greater than 0.0 are accepted")
        iteration = 0
        
        self.partition_initialization(self, b)
        
        while self.k <= self.target_k:
            self.certs_compls_list = []
            self.regularity_list = []
            iteration += 1
            num_of_irregular_pairs = self.check_pairs_regularity()
            if num_of_irregular_pairs == -1:
                return (False, self.k, self.classes, self.sze_idx, self.regularity_list, len(self.irr_list), self.irr_list)
            #ipdb.set_trace()
            if self.k >= max_k:
                if verbose:
                    # Cardinality too low irregular by definition
                    print("{0}, {1}, {2}, {3}, {4}, irregular [cardinality too low]".format(iteration, self.k, self.classes_cardinality, self.sze_idx,num_of_irregular_pairs))
                return (False, self.k, None, self.sze_idx, self.regularity_list, num_of_irregular_pairs, self.irr_list)
                
            if self.check_partition_regularity(num_of_irregular_pairs):
                if self.k == self.target_k:
                    if verbose:
                        print("{0}, {1}, {2}, {3}, {4}, regular [k = target_k]".format(iteration, self.k, self.classes_cardinality, self.sze_idx, num_of_irregular_pairs))
                    return (True, self.k, self.classes, self.sze_idx, self.regularity_list, len(self.irr_list), self.irr_list)
                else:
                    if verbose:
                        print("{0}, {1}, {2}, {3}, {4}, regular [k < target_k]".format(iteration, self.k, self.classes_cardinality, self.sze_idx, num_of_irregular_pairs))
            else:
                if verbose:
                    print("{0}, {1}, {2}, {3}, {4}, irregular [too many irregular pairs]".format(iteration, self.k, self.classes_cardinality, self.sze_idx, num_of_irregular_pairs) )
            
            if self.k == self.target_k:
                break
            res = self.refinement_step(self)
            if not res:
                if verbose:
                    # Irregular by definition
                    print("{0}, {1}, {2}, {3}, {4}, irregular [by definition]".format(iteration, self.k, self.classes_cardinality, self.sze_idx, num_of_irregular_pairs))
                return (False, self.k, None, self.sze_idx, self.regularity_list, num_of_irregular_pairs, self.irr_list)
            
        #self.generate_reduced_sim_mat()
        #return (True, self.k, self.reduced_sim_mat)
        #ipdb.set_trace()
        return (False, self.k, self.classes, self.sze_idx, self.regularity_list, len(self.irr_list), self.irr_list)
