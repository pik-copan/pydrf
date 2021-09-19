from sensible_raw.loaders import loader
from world_viewer.cns_world import CNSWorld
from world_viewer.glasses import Glasses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LogNorm
from sklearn.utils import shuffle
import pickle
from itertools import groupby 

def shuffle_net(a_ij, n_steps):
    active = a_ij.loc[a_ij.edge == True]
    passive = a_ij.loc[a_ij.edge == False]
    
    
    for u in range(n_steps):
        
        # choose first edge (i,j)
        i,j = active.sample(1)[["id_A","id_B"]].values[0]
        
        # choose first non-edge (j,k)
        non_neighb_j = passive.loc[(passive.id_A == j) | (passive.id_B == j)]
        if len(non_neighb_j) == 0: continue
        jk = non_neighb_j.sample(1)
        k = jk.id_B.values[0]
        jk_swap = False
        if k == j: 
            jk_swap = True
            k = jk.id_A.values[0]
        
        # choose second edge (k,l)        
        possible_l = list(active.loc[(active.id_A == k) & (active.id_B != i)  , 'id_B'].values)
        possible_l += list(active.loc[(active.id_B == k) & (active.id_A != i)  , 'id_A'].values)
        if len(possible_l) == 0: continue
            
        # choose second non-edge
        candidates = list(passive.loc[(passive.id_A == i), 'id_B'].values)
        candidates += list(passive.loc[(passive.id_B == i), 'id_A'].values)
        l_candidates = [value for value in possible_l if value in set(candidates)] 
        if len(l_candidates) == 0: continue
        l = l_candidates[np.random.randint(len(l_candidates))]
        
        # rewire edges
        a_ij.set_index(['id_A','id_B'],inplace=True)
        a_ij.loc[(i,j),"edge"] = 0
        if jk_swap:
            a_ij.loc[(k,j),"edge"] = 1
        else:
            a_ij.loc[(j,k),"edge"] = 1
            
        try:
            a_ij.loc[(k,l),"edge"] = 0
        except KeyError:
            a_ij.loc[(l,k),"edge"] = 0
            
        try:
            a_ij.loc[(i,l),"edge"] = 1
        except KeyError:
            a_ij.loc[(l,i),"edge"] = 1
            
        a_ij.reset_index(inplace=True)
        
    return a_ij

path = "tmp/ShuffledV4"


# shuffle time
for run in range(0,5):
    # load data
    cns = CNSWorld()
    cns.load_world(opinions = ['fitness'], read_cached = True, stop=False, write_pickle = False, continous_op=False)
    cns.d_ij = None
    cns_glasses = Glasses(cns)
    
    #time restriction
    start_spring = "2014-01-15"
    end_spring = "2014-04-30"
    cns.time = cns.time.loc[(cns.time.time >= start_spring) & (cns.time.time <= end_spring)]
    cns.op_nodes = cns.op_nodes.loc[(cns.op_nodes.time >= start_spring) & (cns.op_nodes.time <= end_spring)]
    cns.a_ij = cns.a_ij.loc[(cns.a_ij.time >= start_spring) & (cns.a_ij.time <= end_spring)]
       
        
    cns.a_ij = cns.a_ij.groupby("time", group_keys=False).apply(shuffle_net,1200).dropna()
    
    cns.a_ij.to_pickle(f"{path}/a_ij_shuffled_edges_cons_indiv_degree_{run}.pkl")

    exposure = cns_glasses.calc_exposure("expo_frac", "op_fitness", exposure_time = 7)
    exposure.to_pickle(f"{path}/exposure_fitness_7_1_shuffled_edges_cons_indiv_degree_{run}.pkl")