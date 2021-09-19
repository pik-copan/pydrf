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

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

path = "tmp/ShuffledAVM"

phi = 0.0
steps = 89

# shuffle time
for run in range(1,11):
    # load data
    avm = SyntheticWorld(path="data/Synthetisch/avm_final_5k", run=1, number_of_nodes=851)
    avm.load_world(phi = phi, cc = False, n_op = 2, steps=steps, read_cached = False, tc=True, load_full_connection_df = True) 
    avm.d_ij = None 
    avm_glasses = Glasses(avm, verbose=True)

    #shuffle time
    avm.a_ij.edge = shuffle(avm.a_ij.edge.values)
    avm.a_ij = avm.a_ij[avm.a_ij["edge"]==True]
    avm.a_ij.to_pickle(f"{path}/a_ij_shuffled_time_complete_phi{phi}_{run}.pkl")

    exposure = avm_glasses.calc_exposure("expo_frac", "op_synthetic", exposure_time = 7)
    exposure.to_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_edges_complete_phi{phi}_{run}.pkl")