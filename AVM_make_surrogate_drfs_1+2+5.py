from world_viewer.synthetic_world import SyntheticWorld
from world_viewer.glasses import Glasses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LogNorm
from sklearn.utils import shuffle
from matplotlib.figure import figaspect

# set analysis parameters
analysis = 'expo_nmb'
opinion_type = "op_synthetic"
binning = True
n_bins = 15
save_plots = False
show_plot = True
phi = 0.0
steps = 89
run=1

# Function for loading Surrogate data
def load_data(runs, mode, analysis = "expo_nmb", path= "tmp/ShuffledAVM", shuffle_type="complete"):
    syn = SyntheticWorld(path="data/Synthetisch/avm_final_5k", run=1, number_of_nodes=851)
    syn.load_world(phi = phi, cc = False, n_op = 2, steps=steps, read_cached = False, tc=True) 
    syn_glasses = Glasses(syn)
    syn_glasses.output_folder = ""
    syn.d_ij = None

    data_all = []
    expo_all = []

    for run in runs:
        # load
        if mode == "edges":
            exposure = pd.read_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_edges_{shuffle_type}_phi{phi}_{run}.pkl")
            syn.a_ij = pd.read_pickle(f"{path}/a_ij_shuffled_time_{shuffle_type}_phi{phi}_{run}.pkl")
        elif mode == "traits":
            exposure = pd.read_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_traits_{shuffle_type}_phi{phi}_{run}.pkl")
            syn.op_nodes = pd.read_pickle(f"{path}/op_nodes_shuffled_time_{shuffle_type}_phi{phi}_{run}.pkl")
        if analysis == "expo_nmb":
            exposure.rename(columns={"exposure":"exposure_old", "n_influencer_summed":"exposure"},inplace=True)
        #restrict
        exposure.reset_index(inplace=True)
        exposure.set_index(["node_id","time"], inplace=True)
        # degree filter
        degree = exposure.groupby("node_id").n_nbs.mean().to_frame("avg").reset_index()
        exposure = exposure.loc[degree.loc[degree.avg >= 4,"node_id"]]
        exposure = exposure[exposure.n_nbs_mean > 1/7]
        #calculate
        data, expo_agg = syn_glasses.opinion_change_per_exposure(exposure, opinion_type, opinion_change_time = 1)
        data_all += [data]
        expo_all += [expo_agg]
    return pd.concat(data_all), pd.concat(expo_all), syn_glasses

# 5. Surrogate
data_edges, expo_all_edges, syn_glasses_edges = load_data(runs=range(4,5), mode="edges", analysis = "expo_nmb", path= "tmp/ShuffledAVM")
data_edges.to_pickle("tmp/final/surrogate5_phi"+str(phi)+".pkl")

# 1. Surrogate
data_traits, expo_all_traits, syn_glasses_traits = load_data(runs=range(1,11), mode="traits", analysis = "expo_nmb", path= "tmp/ShuffledAVM")
data_traits.to_pickle("tmp/final/surrogate1_phi"+str(phi)+".pkl")

# 2. Surrogate
data_traits_pn, expo_all_traits_pn, syn_glasses_traits_pn = load_data(runs=range(1,11), mode="traits", analysis = "expo_nmb", path= "tmp/ShuffledAVM", shuffle_type="complete_per_node")
data_traits_pn.to_pickle("tmp/final/surrogate2_phi"+str(phi)+".pkl")

