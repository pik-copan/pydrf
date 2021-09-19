# from sensible_raw.loaders import loader
from world_viewer.synthetic_world import SyntheticWorld
from world_viewer.glasses import Glasses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LogNorm
from sklearn.utils import shuffle
import pickle
from itertools import groupby 

def intervall_shuffle(shuffle_list):
    groups = [list(y) for x, y in groupby(shuffle_list)] 
    groups[::2] = shuffle(groups[::2]) #even
    groups[1::2] = shuffle(groups[1::2]) #odd
    res = [item for group in groups for item in group]
    return res

path = "tmp/ShuffledAVM"

phi = 0.0
steps = 89

# shuffle time
for run in range(1,11):
    # load data
    avm = SyntheticWorld(path="data/Synthetisch/avm_final_5k", run=1, number_of_nodes=851)
    avm.load_world(phi = phi, cc = False, n_op = 2, steps=steps, read_cached = False, tc=True) 
    avm.d_ij = None 
    avm_glasses = Glasses(avm, verbose=True)
       
    avm.op_nodes.set_index("node_id",inplace=True)
    avm.op_nodes.sort_index(inplace=True)
    avm.op_nodes["op_synthetic"] = avm.op_nodes.groupby("node_id").op_synthetic.apply(
                lambda o: pd.DataFrame(intervall_shuffle(o.values), columns=["op_synthetic"]))\
                .reset_index().set_index("node_id").op_synthetic
    avm.op_nodes.reset_index(inplace=True)
    avm.op_nodes.to_pickle(f"{path}/op_nodes_shuffled_time_complete_per_node_conserved_switches_phi{phi}_{run}.pkl")  

    exposure = avm_glasses.calc_exposure("expo_frac", "op_synthetic", exposure_time = 7)
    exposure.to_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_traits_complete_per_node_conserved_switches_phi{phi}_{run}.pkl")