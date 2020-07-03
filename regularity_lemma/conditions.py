# Source: https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions/blob/master/graph_reducer/

import numpy as np
import scipy.sparse.linalg
import math
import sys
#import ipdb


def alon1(self, cl_pair):
    """
    verify the first condition of Alon algorithm (regularity of pair)
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    :return: A list of two empty lists representing the empty certificates
    :return: A list of two empty lists representing the empty complements
    """
    return cl_pair.bip_avg_deg < (self.epsilon ** 3.0) * cl_pair.classes_n, [[], []], [[], []]


def alon2(self, cl_pair):
    """ Verifies the third condition of Alon algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """

    # Gets the vector of degrees of nodes of class s wrt class r
    s_degrees = cl_pair.s_r_degrees[cl_pair.s_indices]

    deviated_nodes_mask = np.abs(s_degrees - cl_pair.bip_avg_deg) >= (self.epsilon ** 4.0) * cl_pair.classes_n

    if deviated_nodes_mask.sum() > (1/8 * self.epsilon**4 * cl_pair.classes_n):
        # [TODO] Heuristic? Zip?
        s_certs = cl_pair.s_indices[deviated_nodes_mask]
        s_compls = np.setdiff1d(cl_pair.s_indices, s_certs)

        # Takes all the indices of class r which are connected to s_certs
        b_mask = self.adj_mat[np.ix_(s_certs, cl_pair.r_indices)] > 0
        b_mask = b_mask.any(0)

        r_certs = cl_pair.r_indices[b_mask]
        r_compls = np.setdiff1d(cl_pair.r_indices, r_certs)

        is_irregular = True
        return is_irregular, [r_certs.tolist(), s_certs.tolist()], [r_compls.tolist(), s_compls.tolist()]
    else:
        is_irregular = False
        return is_irregular, [[], []], [[], []]


def alon3(self, cl_pair):
    """ Verifies the third condition of Alon algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """
    is_irregular = False

    nh_dev_mat = cl_pair.neighbourhood_deviation_matrix()

    # Gets the vector of degrees of nodes of class s wrt class r
    s_degrees = cl_pair.s_r_degrees[cl_pair.s_indices]

    yp_filter = cl_pair.find_Yp(s_degrees, cl_pair.s_indices)

    if yp_filter.size == 0:
        is_irregular = True
        return is_irregular, [[], []], [[], []]

    s_certs, y0 = cl_pair.compute_y0(nh_dev_mat, cl_pair.s_indices, yp_filter)

    if s_certs is None:
        is_irregular = False
        return is_irregular, [[], []], [[], []]
    else:
        assert np.array_equal(np.intersect1d(s_certs, cl_pair.s_indices), s_certs) == True, "cert_is not subset of s_indices"
        assert (y0 in cl_pair.s_indices) == True, "y0 not in s_indices"

        is_irregular = True
        b_mask = self.adj_mat[np.ix_(np.array([y0]), cl_pair.r_indices)] > 0
        r_certs = cl_pair.r_indices[b_mask[0]]
        assert np.array_equal(np.intersect1d(r_certs, cl_pair.r_indices), r_certs) == True, "cert_is not subset of s_indices"

        # [BUG] cannot do set(s_indices) - set(s_certs)
        s_compls = np.setdiff1d(cl_pair.s_indices, s_certs)
        r_compls = np.setdiff1d(cl_pair.r_indices, r_certs)
        assert s_compls.size + s_certs.size == self.classes_cardinality, "Wrong cardinality"
        assert r_compls.size + r_certs.size == self.classes_cardinality, "Wrong cardinality"

        return is_irregular, [r_certs.tolist(), s_certs.tolist()], [r_compls.tolist(), s_compls.tolist()]

def frieze_kannan(self, cl_pair):
    """
    verify the condition of Frieze and Kannan algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """
    cert_r = []
    cert_s = []
    compl_r = []
    compl_s = []

    if self.is_weighted:
        W = cl_pair.bip_sim_mat - cl_pair.bip_density
    else:
        W = cl_pair.bip_adj_mat - cl_pair.bip_density

    x, sv_1, y = scipy.sparse.linalg.svds(W, k=1, which='LM')

    is_irregular = (sv_1 >= self.epsilon * cl_pair.n)

    if is_irregular:
        beta = 3.0 / self.epsilon
        x = x.ravel()
        y = y.ravel()
        hat_thresh = beta / math.sqrt(cl_pair.n)
        x_hat = np.where(np.abs(x) <= hat_thresh, x, 0.0)
        y_hat = np.where(np.abs(y) <= hat_thresh, y, 0.0)

        quadratic_threshold = (self.epsilon - 2.0 / beta) * (cl_pair.n / 4.0)

        x_mask = x_hat > 0
        y_mask = y_hat > 0
        x_plus = np.where(x_mask, x_hat, 0.0)
        x_minus = np.where(~x_mask, x_hat, 0.0)
        y_plus = np.where(y_mask, y_hat, 0.0)
        y_minus = np.where(~y_mask, y_hat, 0.0)

        r_mask = np.empty((0, 0))
        s_mask = np.empty((0, 0))

        q_plus = y_plus * 1.0 / hat_thresh
        q_minus = y_minus * 1.0 / hat_thresh

        if x_plus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) >= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_plus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) >= 0.0
            s_mask = (r_mask @ W) <= 0.0
        elif x_minus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) <= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_minus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) <= 0.0
            s_mask = (r_mask @ W) <= 0.0
        else:
            sys.exit("no condition on the quadratic form was verified")

        cert_r = list(cl_pair.index_map[0][r_mask])
        compl_r = list(cl_pair.index_map[0][~r_mask])
        cert_s = list(cl_pair.index_map[1][s_mask])
        compl_s = list(cl_pair.index_map[1][~s_mask])
    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]