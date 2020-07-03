# Source: https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions/blob/master/graph_reducer/

import numpy as np
import random
import sys
#import ipdb
import logging


def random_based(self):
    """ Perform step 4 of Alon algorithm, performing the refinement of the pairs, processing nodes in a random way. Some heuristic is applied in order to speed up the process.
    """
    pass


def partition_correct(self):
    """ Checks if the partition cardinalities are valid
    :returns: True if the classes of the partition have the right cardinalities, false otherwise
    """
    for i in range(1, self.k+1):
        if not np.where(self.classes == i)[0].size == self.classes_cardinality:
            return False
    return True



##########################################################################
######################## INDEGREE REFINEMENT #############################
##########################################################################


def density(self, indices_a, indices_b):
    """ Calculates the density between two sets of vertices
    :param indices_a: np.array(), the indices of the first set
    :param indices_b: np.array(), the indices of the second set
    """
    if indices_a.size == 0 or indices_b.size == 0:
        return 0
    elif indices_a.size == indices_b.size == 1:
        return 0

    # [TODO] performance issue: comparing all the indices? maybe add a parameter to the function
    if np.array_equal(indices_a, indices_b):
        n = indices_a.size
        max_edges = (n*(n-1))/2
        n_edges = np.tril(self.adj_mat[np.ix_(indices_a, indices_a)], -1).sum()
        #n_edges = np.tril(self.sim_mat[np.ix_(indices_a, indices_a)], -1).sum()
        return n_edges / max_edges

    n_a = indices_a.size
    n_b = indices_b.size
    max_edges = n_a * n_b
    n_edges = self.adj_mat[np.ix_(indices_a, indices_b)].sum()
    #n_edges = self.sim_mat[np.ix_(indices_a, indices_b)].sum()
    return n_edges / max_edges


def compute_indensities(self):
    """ Compute the inside density for each class of a given partition
    :returns: np.array(float32) of densities for each class in the partition
    """
    cls = list(range(0, self.k + 1))
    densities = np.zeros(len(cls), dtype='float32')
    for c in cls:
        c_indices = np.where(self.classes == c)[0]
        if c_indices.size:
            densities[c] = density(self, c_indices, c_indices)
        else:
            densities[c] = 0

    return densities


def choose_candidate(self, in_densities, s, irregulars):
    """ This function chooses a class between the irregular ones (d(ci,cj), 1-|d(ci,ci)-d(cj,cj)|)
    :param in_densities: list(float), precomputed densities to speed up the calculations
    :param s: int, the class which all the other classes are compared to
    :param irregulars: list(int), the list of irregular classes
    """
    candidate_idx = -1
    candidate = -1

    # Exploit the precalculated densities
    s_dens = in_densities[s]
    for r in irregulars:
        s_indices = np.where(self.classes == s)[0]
        r_indices = np.where(self.classes == r)[0]
        r_idx = density(self, s_indices, r_indices) + (1 - abs(s_dens - in_densities[r]))
        if r_idx > candidate_idx:
            candidate_idx = r_idx
            candidate = r

    return candidate


def fill_new_set(self, new_set, compls, maximize_density):
    """ Find nodes that can be added
    Move from compls the nodes in can_be_added until we either finish the nodes or reach the desired cardinality
    :param new_set: np.array(), array of indices of the set that must be augmented
    :param compls: np.array(), array of indices used to augment the new_set
    :param maximize_density: bool, used to augment or decrement density
    """

    if maximize_density:
        nodes = self.adj_mat[np.ix_(new_set, compls)] == 1.0
        #nodes = self.sim_mat[np.ix_(new_set, compls)] >= 0.5

        # These are the nodes that can be added to certs, we take the most connected ones with all the others
        to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        to_add = to_add[0][to_add[1].argsort()]
    else:
        nodes = self.adj_mat[np.ix_(new_set, compls)] == 0.0
        #nodes = self.sim_mat[np.ix_(new_set, compls)] < 0.5

        # These are the nodes that can be added to certs, we take the less connected ones with all the others
        to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        to_add = to_add[0][to_add[1].argsort()[::-1]]

    while new_set.size < self.classes_cardinality:

        # If there are nodes in to_add, we keep moving from compls to new_set
        if to_add.size > 0:
            node, to_add = to_add[-1], to_add[:-1]
            new_set = np.append(new_set, node)
            compls = np.delete(compls, np.argwhere(compls == node))
        else:
            # If there aren't candidate nodes, we keep moving from complements
            # to certs until we reach the desired cardinality
            node, compls = compls[-1], compls[:-1]
            new_set = np.append(new_set, node)

    return new_set, compls


def indeg_guided(self):
    """ In-degree based refinement. The refinement exploits the internal structure of the classes of a given partition.
    :returns: True if the new partition is valid, False otherwise
    """
    #ipdb.set_trace()
    threshold = 0.5

    to_be_refined = list(range(1, self.k + 1))
    old_cardinality = self.classes_cardinality
    self.classes_cardinality //= 2
    in_densities = compute_indensities(self)
    new_k = 0

    while to_be_refined:
        #print("Il nuovo k " + str(new_k))
        s = to_be_refined.pop(0)
        irregular_r_indices = []

        for r in to_be_refined:
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        # If class s has irregular classes
        if irregular_r_indices:

            # Choose candidate based on the inside-outside density index
            r = choose_candidate(self, in_densities, s, irregular_r_indices)
            to_be_refined.remove(r)

            s_certs = np.array(self.certs_compls_list[r - 2][s - 1][0][1]).astype('int32')
            s_compls = np.array(self.certs_compls_list[r - 2][s - 1][1][1]).astype('int32')
            assert s_certs.size + s_compls.size == old_cardinality

            r_compls = np.array(self.certs_compls_list[r - 2][s - 1][1][0]).astype('int32')
            r_certs = np.array(self.certs_compls_list[r - 2][s - 1][0][0]).astype('int32')
            assert r_certs.size + r_compls.size == old_cardinality


            # Merging the two complements
            compls = np.append(s_compls, r_compls)

            # Calculating certificates densities
            dens_s_cert = density(self, s_certs, s_certs)
            dens_r_cert = density(self, r_certs, r_certs)
            
            for cert, dens in [(s_certs, dens_s_cert), (r_certs, dens_r_cert)]:

                # Indices of the cert ordered by in-degree, it doesn't matter if we reverse the list as long as we unzip it
                degs = self.adj_mat[np.ix_(cert, cert)].sum(1).argsort()[::-1]
                #degs = self.sim_mat[np.ix_(cert, cert)].sum(1).argsort()[::-1]

                if dens > threshold:
                    # Certificates high density branch
                    # Unzip them in half to preserve seeds
                    set1=  cert[degs[0:][::2]]
                    set2 =  cert[degs[1:][::2]]

                    # Adjust cardinality of the new set to the desired cardinality
                    set1, compls = fill_new_set(self, set1, compls, True)
                    set2, compls = fill_new_set(self, set2, compls, True)

                    # Handling of odd classes
                    new_k -= 1
                    self.classes[set1] = new_k
                    if set1.size > self.classes_cardinality:
                        self.classes[set1[-1]] = 0
                    new_k -= 1
                    self.classes[set2] = new_k
                    if set2.size > self.classes_cardinality:
                        self.classes[set2[-1]] = 0

                else:
                    # Certificates low density branch
                    set1 = np.random.choice(cert, len(cert)//2, replace=False)
                    set2 = np.setdiff1d(cert, set1)

                    # Adjust cardinality of the new set to the desired cardinality
                    set1, compls = fill_new_set(self, set1, compls, False)
                    set2, compls = fill_new_set(self, set2, compls, False)

                    # Handling of odd classes
                    new_k -= 1
                    self.classes[set1] = new_k
                    if set1.size > self.classes_cardinality:
                        self.classes[set1[-1]] = 0
                    new_k -= 1
                    self.classes[set2] = new_k
                    if set2.size > self.classes_cardinality:
                        self.classes[set2[-1]] = 0

                # Handle special case when there are still some complements not assigned
                if compls.size > 0:
                    self.classes[compls] = 0

        else:
            # The class is e-reg with all the others or it does not have irregular classes
            # Sort by indegree and unzip the structure
            s_indices = np.where(self.classes == s)[0]
            s_indegs = self.adj_mat[np.ix_(s_indices, s_indices)].sum(1).argsort()
            #s_indegs = self.sim_mat[np.ix_(s_indices, s_indices)].sum(1).argsort()

            set1=  s_indices[s_indegs[0:][::2]]
            set2=  s_indices[s_indegs[1:][::2]]

            # Handling of odd classes
            new_k -= 1
            self.classes[set1] = new_k
            if set1.size > self.classes_cardinality:
                self.classes[set1[-1]] = 0
            new_k -= 1
            self.classes[set2] = new_k
            if set1.size > self.classes_cardinality:
                self.classes[set1[-1]] = 0

    self.k *= 2

    # Check validity of class C0, if invalid and enough nodes, distribute the exceeding nodes among the classes
    c0_indices = np.where(self.classes == 0)[0]
    if c0_indices.size >= (self.epsilon * self.adj_mat.shape[0]):
        if c0_indices.size > self.k:
            self.classes[c0_indices[:self.k]] = np.array(range(1, self.k+1))*-1
        else:
            print('[ refinement ] Invalid cardinality of C_0')
            return False

    self.classes *= -1

    if not partition_correct(self):
        ipdb.set_trace()
    return True


##########################################################################
######################## PAIR DEGREE REFINEMENT ##########################
##########################################################################


def within_degrees(self, c):
    """ Given a class c it returns the degrees calculated within the class
    :param c: int, class c
    :returns: np.array(int16), list of n indices where the indices in c have the in-degree
    """
    c_degs = np.zeros(len(self.degrees), dtype='int16')
    c_indices = np.where(self.classes == c)[0]
    c_degs[c_indices] = self.adj_mat[np.ix_(c_indices, c_indices)].sum(1)

    return c_degs


def get_s_r_degrees(self,s,r):
    """ Given two classes it returns a degree vector (indicator vector) where the degrees
    have been calculated with respecto to each other set.
    :param s: int, class s
    :param r: int, class r
    :returns: np.array, degree vector
    """

    s_r_degs = np.zeros(len(self.degrees), dtype='int16')

    # Gets the indices of elements which are part of class s, then r
    s_indices = np.where(self.classes == s)[0]
    r_indices = np.where(self.classes == r)[0]

    # Calculates the degree and assigns it
    s_r_degs[s_indices] = self.adj_mat[np.ix_(s_indices, r_indices)].sum(1)
    s_r_degs[r_indices] = self.adj_mat[np.ix_(r_indices, s_indices)].sum(1)

    return s_r_degs



def degree_based(self):
    """
    perform step 4 of Alon algorithm, performing the refinement of the pairs, processing nodes according to their degree. Some heuristic is applied in order to
    speed up the process
    """
    #ipdb.set_trace()
    to_be_refined = list(range(1, self.k + 1))
    irregular_r_indices = []
    is_classes_cardinality_odd = self.classes_cardinality % 2 == 1
    self.classes_cardinality //= 2

    while to_be_refined:
        #print("Ecco il k refine "+ str(self.k))
        s = to_be_refined.pop(0)

        for r in to_be_refined:
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        if irregular_r_indices:
            #np.random.seed(314)
            #random.seed(314)
            chosen = random.choice(irregular_r_indices)
            to_be_refined.remove(chosen)
            irregular_r_indices = []

            # Degrees wrt to each other class
            s_r_degs = get_s_r_degrees(self, s, chosen)

            # i = 0 for r, i = 1 for s
            for i in [0, 1]:
                cert_length = len(self.certs_compls_list[chosen - 2][s - 1][0][i])
                compl_length = len(self.certs_compls_list[chosen - 2][s - 1][1][i])

                greater_set_ind = np.argmax([cert_length, compl_length])
                lesser_set_ind = np.argmin([cert_length, compl_length]) if cert_length != compl_length else 1 - greater_set_ind

                greater_set = self.certs_compls_list[chosen - 2][s - 1][greater_set_ind][i]
                lesser_set = self.certs_compls_list[chosen - 2][s - 1][lesser_set_ind][i]

                self.classes[lesser_set] = 0

                difference = len(greater_set) - self.classes_cardinality
                # retrieve the first <difference> nodes sorted by degree.
                # N.B. NODES ARE SORTED IN DESCENDING ORDER
                difference_nodes_ordered_by_degree = sorted(greater_set, key=lambda el: s_r_degs[el], reverse=True)[0:difference]
                #difference_nodes_ordered_by_degree = sorted(greater_set, key=lambda el: np.where(self.degrees == el)[0], reverse=True)[0:difference]

                self.classes[difference_nodes_ordered_by_degree] = 0
        else:
            self.k += 1
            #  TODO: cannot compute the r_s_degs since the candidate does not have any e-regular pair  <14-11-17, lakj>
            s_indices_ordered_by_degree = sorted(list(np.where(self.classes == s)[0]), key=lambda el: np.where(self.degrees == el)[0], reverse=True)
            #s_indices_ordered_by_degree = sorted(list(np.where(self.classes == s)[0]), key=lambda el: s_r_degs[el], reverse=True)

            if is_classes_cardinality_odd:
                self.classes[s_indices_ordered_by_degree.pop(0)] = 0
            self.classes[s_indices_ordered_by_degree[0:self.classes_cardinality]] = self.k

    C0_cardinality = np.sum(self.classes == 0)
    num_of_new_classes = C0_cardinality // self.classes_cardinality
    nodes_in_C0_ordered_by_degree = np.array([x for x in self.degrees if x in np.where(self.classes == 0)[0]])
    for i in range(num_of_new_classes):
        self.k += 1
        self.classes[nodes_in_C0_ordered_by_degree[
                     (i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = self.k

    C0_cardinality = np.sum(self.classes == 0)
    if C0_cardinality > self.epsilon * self.N:
        #sys.exit("Error: not enough nodes in C0 to create a new class.Try to increase epsilon or decrease the number of nodes in the graph")
        #print("Error: not enough nodes in C0 to create a new class. Try to increase epsilon or decrease the number of nodes in the graph")

        if not partition_correct(self):
            ipdb.set_trace()
        return False

    if not partition_correct(self):
        ipdb.set_trace()
    return True
