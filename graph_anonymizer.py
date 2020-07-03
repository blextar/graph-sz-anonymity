# Authors: Luca Rossi, Giorgia Minello, and Daniele Foffano
# Reference: k-Anonymity on Graphs using the Szemere ́di Regularity Lemma (under review)

import anonymization_functions as af
import networkx as nx
import numpy as np
import sys
import time
import os
import operator
import getopt

def graph_anonymizer(graph_name,n,heuristic="indeg_guided",verbose=False):
    
    l = 2**n
    
    l2k = {}
    l2k['PT'] = {4:478,8:239,16:120,32:60,64:30,128:14,256:7}
    l2k['TV'] = {4:973,8:487,16:243,32:122,64:61,128:30,256:15}
    l2k['FB'] = {4:1010,8:505,16:252,32:126,64:63,128:31,256:15}
    l2k['ES'] = {4:1162,8:581,16:290,32:145,64:73,128:36,256:18}
    l2k['POL'] = {4:1477,8:739,16:369,32:185,64:92,128:46,256:23}
    l2k['GOV'] = {4:1764,8:882,16:441,32:221,64:110,128:55,256:27}
    l2k['ART'] = {4:12629,8:6314,16:3157,32:1579,64:789,128:395,256:197}
    l2k['GP'] = {4:26903,8:13452}

    G_init = nx.read_edgelist("test_graphs/" + graph_name + ".txt",nodetype=int)
    is_directed = nx.is_directed(G_init)
    G_init = G_init.to_undirected()    
 
    # add enough nodes to ensure that |C0| = 0
    number_of_nodes = nx.number_of_nodes(G_init)
    C0_size = l-number_of_nodes%l
    G_init.add_nodes_from(list(range(number_of_nodes,number_of_nodes+C0_size)))

    total_iterations = 50
    max_attempts = 100 

    runtimes = list()
    epsilons = list()
    num_irr_pairs = list()
    nodes_in_irr_pairs = list()
        
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log = open(results_dir + "/" + graph_name + ".txt.k" + str(l2k[graph_name][l]) + ".sz.txt.log",'w')
    sys.stdout = log
    
    for iteration in range(1,total_iterations+1):
        print("### Iteration number " + str(iteration) + " ###")
        start_time = time.time()
    
        matrix =  np.asarray(nx.to_numpy_matrix(G_init,nodelist=range(nx.number_of_nodes(G_init))))
          
        returned_values = af.anonymize(matrix, l, is_directed, heuristic, verbose=verbose)
        attempt = 0
        while returned_values == None and attempt < max_attempts:
            start_time = time.time()
            
            matrix =  np.asarray(nx.to_numpy_matrix(G_init,nodelist=range(nx.number_of_nodes(G_init))))
            
            returned_values = af.anonymize(matrix, l, is_directed, heuristic, verbose=verbose)
            attempt += 1
        if attempt == max_attempts and returned_values == None:
            print("Failure: max_attempts reached")
            sys.exit(2)
    
        matrix_an, epsilon, sze_idx, irr_pairs, irr_list, classes = returned_values
                
        stop_time = time.time()
        elapsed_time = stop_time-start_time
        runtimes.append(elapsed_time)
        epsilons.append(epsilon)
        num_irr_pairs.append(irr_pairs)
    
        G = nx.from_numpy_matrix(matrix_an)
        nx.write_adjlist(G,results_dir + "/" + graph_name + ".txt.k" + str(l2k[graph_name][l]) + ".i" + str(iteration) + ".sz.txt")
        np.savetxt(results_dir + "/" + graph_name + ".txt.k" + str(l2k[graph_name][l]) + ".i" + str(iteration) + ".sz.txt.node_membership.txt",classes,fmt='%d')  
        print("Iteration completed\n")
        
    with open(results_dir + "/" + graph_name + ".txt.k" + str(l2k[graph_name][l]) + ".sz.txt.epsilons.txt", 'w') as f:
        print(epsilons, file=f)
    with open(results_dir + "/" + graph_name + ".txt.k" + str(l2k[graph_name][l]) + ".sz.txt.runtimes.txt", 'w') as f:
        print(runtimes, file=f)
    with open(results_dir + "/" + graph_name + ".txt.k" + str(l2k[graph_name][l]) + ".sz.txt.num_irr_pairs.txt", 'w') as f:
        print(num_irr_pairs, file=f)
        
def main(argv):
    
    graph_name = None
    n = None
    
    try:
        opts, args = getopt.getopt(argv,"g:n:")
    except getopt.GetoptError:
        sys.exit("graph_anonymizer.py -g <graph_name> -n <2^n>")
    for opt, arg in opts:
        if opt == '-g':
            graph_name = arg
        elif opt == '-n':
            try:
                n = int(arg)
            except ValueError:
                sys.exit("n should be an integer greater than or equal to 2")
            if n < 2:
                sys.exit("n should be an integer greater than or equal to 2")
    
    if graph_name is None:
        print("Please specify a graph name: [PT | TV | FB | ES | POL | GOV]")
        sys.exit("Syntax: graph_anonymizer.py -g <graph_name> -n <2^n>")
        
    if n is None:
        n = 2
        print("Using default n = 2")
        
    graph_anonymizer(graph_name,n,verbose=True)

if __name__ == "__main__":
    main(sys.argv[1:])
