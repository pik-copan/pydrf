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

run=1
phi=0.6
steps = 255

# loader for surrogate data
def load_data(runs, analysis = "expo_frac", path= "tmp/ShuffledAVM"):
    syn = SyntheticWorld(path="data/Synthetisch/avm_final_5k", run=1, number_of_nodes=851)
    syn.load_world(phi = phi, cc = False, n_op = 2, steps=steps, read_cached = False, tc=True) 
    syn_glasses = Glasses(syn)
    syn_glasses.output_folder = ""
    syn.d_ij = None
    
    data_all = []
    expo_all = []

    for run in runs:
        # load
        exposure = pd.read_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_traits_complete_per_node_conserved_switches_phi{phi}_{run}.pkl")
        syn.op_nodes = pd.read_pickle(f"{path}/op_nodes_shuffled_time_complete_per_node_conserved_switches_phi{phi}_{run}.pkl")
        if analysis == "expo_nmb":
            exposure.rename(columns={"exposure":"exposure_old", "n_influencer_summed":"exposure"},inplace=True)
        #restrict
        exposure.reset_index(inplace=True)
        # exposure = exposure.loc[(exposure.time >= pd.to_datetime(start)) & (exposure.time <= pd.to_datetime(end))]
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

# load data
data_traits, expo_traits, syn_glasses = load_data(range(1,11), analysis = "expo_nmb")
data_traits.to_pickle("tmp/final/surrogate3_phi"+str(phi)+".pkl")