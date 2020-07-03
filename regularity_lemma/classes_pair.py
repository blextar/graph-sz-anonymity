# Source: https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions/blob/master/graph_reducer/

import numpy as np
#import ipdb


class ClassesPair:

    def __init__(self, adj_mat, classes, r, s, epsilon):

        # Classes id
        self.r = r
        self.s = s

        # Classes indices w.r.t. original graph uint16 from 0 to 65535
        self.s_indices = np.where(classes == self.s)[0].astype('uint16')
        self.r_indices = np.where(classes == self.r)[0].astype('uint16')

        # Bipartite adjacency matrix, we've lost the indices w.r.t. adj_mat, inherits dtype of adj_mat
        self.bip_adj_mat = adj_mat[np.ix_(self.s_indices, self.r_indices)]

        # Cardinality of the classes
        self.classes_n = self.bip_adj_mat.shape[0]

        # Bipartite average degree
        self.bip_avg_deg = (self.bip_adj_mat.sum(0) + self.bip_adj_mat.sum(1)).sum() / (2.0 * self.classes_n)

        # Compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the bipartite graph
        self.bip_density = self.bip_adj_mat.sum() / (self.classes_n ** 2.0)

        # Current epsilon used
        self.epsilon = epsilon

        # Degree vector (indicator vector) where the degrees have been calculated with respect to each other set.
        self.s_r_degrees = np.zeros(len(classes), dtype='uint16')

        # Calculates the degree and assigns it
        self.s_r_degrees[self.s_indices] = np.asarray(adj_mat[np.ix_(self.s_indices, self.r_indices)]).sum(1)
        self.s_r_degrees[self.r_indices] = np.asarray(adj_mat[np.ix_(self.r_indices, self.s_indices)]).sum(1)


    def neighbourhood_deviation_matrix(self):

        mat = self.bip_adj_mat.T @ self.bip_adj_mat
        mat = mat.astype('float32')
        mat -= (self.bip_avg_deg ** 2.0) / self.classes_n
        return mat


    def find_Yp(self, bip_degrees, s_indices):
        """ Find a subset of s_indices which will create the Y', it could return an empty array
        :param bip_degrees: np.array(int32) array of the degrees of s nodes w.r.t. class r
        :param s_indices: np.array(int32) array of the indices of class s
        :return: np.array(int32) subset of indices of class s 
        """
        mask = np.abs(bip_degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.classes_n)
        yp_i = np.where(mask == True)[0]
        return yp_i


    def compute_y0(self, nh_dev_mat, s_indices, yp_i):
        """ Finds y0 index node and certificates indices 
        :param nh_dev_mat: np.array((s.size, s.size), dtype='float32') neighbourhood deviation matrix of class s
        :param s_indices: np.array(self.classes_cardinality, dtype='float32') indices of the nodes of class s
        :param yp_i: np.array(dtype='float32') these are indices of the rows to be filtered in the nh_dev_mat
        :return: tuple cert_s which is a np.array(float32) subset of s_indices if it possible, None otherwise

        [TODO] : 
            - why yp_i float32?
            - type of cert_s is float32?
        """

        # Create rectancular matrix to create |y'| sets
        rect_mat = nh_dev_mat[yp_i]

        # Check which set have the best neighbour deviation
        boolean_matrix = rect_mat > (2 * self.epsilon**4 * self.classes_n)
        cardinality_by0s = boolean_matrix.sum(1)

        # Select the best set
        y0_idx = np.argmax(cardinality_by0s)
        aux = yp_i[y0_idx]

        # Gets the y0 index
        y0 = s_indices[aux]

        if cardinality_by0s[y0_idx] > (self.epsilon**4 * self.classes_n / 4.0):
            cert_s = s_indices[boolean_matrix[y0_idx]]
            return cert_s, y0
        else:
            return None, y0


class WeightedClassesPair(ClassesPair):

    def __init__(self, sim_mat, adj_mat, classes, r, s, epsilon):
        # Classes id
        self.r = r
        self.s = s

        # Classes indices w.r.t. original graph uint16 from 0 to 65535
        self.s_indices = np.where(classes == self.s)[0].astype('uint16')
        self.r_indices = np.where(classes == self.r)[0].astype('uint16')

        # Bipartite adjacency matrix, we've lost the indices w.r.t. adj_mat, inherits dtype of adj_mat
        self.bip_adj_mat = adj_mat[np.ix_(self.s_indices, self.r_indices)]
        self.bip_sim_mat = sim_mat[np.ix_(self.s_indices, self.r_indices)]

        # Cardinality of the classes
        self.classes_n = self.bip_adj_mat.shape[0]

        # Bipartite average degree
        self.bip_avg_deg = (self.bip_sim_mat.sum(0) + self.bip_sim_mat.sum(1)).sum() / (2.0 * self.classes_n)

        # Compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the bipartite graph
        self.bip_density = self.bip_sim_mat.sum() / (self.classes_n ** 2.0)

        # Current epsilon used
        self.epsilon = epsilon

        # Degree vector (indicator vector) where the degrees have been calculated with respect to each other set.
        self.s_r_degrees = np.zeros(len(classes), dtype='uint16')

        # Calculates the degree and assigns it
        self.s_r_degrees[self.s_indices] = adj_mat[np.ix_(self.s_indices, self.r_indices)].sum(1)
        self.s_r_degrees[self.r_indices] = adj_mat[np.ix_(self.r_indices, self.s_indices)].sum(1)


