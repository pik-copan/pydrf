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

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

path = "tmp/ShuffledV4"


# shuffle time
for run in range(15,30):
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
    
    cns.op_nodes.op_fitness = shuffle(cns.op_nodes.op_fitness.values)
    cns.op_nodes.to_pickle(f"{path}/op_nodes_shuffled_time_complete_{run}.pkl")  

    exposure = cns_glasses.calc_exposure("expo_frac", "op_fitness", exposure_time = 7)
    exposure.to_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_traits_complete_{run}.pkl")