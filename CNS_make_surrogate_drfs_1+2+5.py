from sensible_raw.loaders import loader
from world_viewer.cns_world import CNSWorld
from world_viewer.glasses import Glasses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LogNorm
from sklearn.utils import shuffle
from matplotlib.figure import figaspect 

# set analysis parameters
analysis = 'expo_frac'
opinion_type = "op_fitness"
binning = True
n_bins = 15
save_plots = False
show_plot = True

# load not shuffeld copenhagen data for comparision
cns_rw = CNSWorld()
cns_rw.load_world(opinions = ['fitness'], read_cached = False, stop=False, write_pickle = False, continous_op=False)
cns_glasses_rw = Glasses(cns_rw)
cns_data_rw = pd.read_pickle("tmp/final/spring_data.pkl")

# Function for loading Surrogate data
def load_data(runs, mode, analysis = "expo_frac", path= "tmp/ShuffledV4", shuffle_type="complete"):
    cns = CNSWorld()
    cns.load_world(opinions = ['fitness'], read_cached = True, stop=False, write_pickle = False, continous_op=False)
    cns_glasses = Glasses(cns)
    cns_glasses.output_folder = ""
    cns.d_ij = None
    
    start = "2014-02-01"
    end = "2014-04-30"
    cns.time = cns.time.loc[(cns.time.time >= start) & (cns.time.time <= end)]
    cns.op_nodes = cns.op_nodes.loc[(cns.op_nodes.time >= start) & (cns.op_nodes.time <= end)]
    cns.a_ij = cns.a_ij.loc[(cns.a_ij.time >= start) & (cns.a_ij.time <= end)]
    
    data_all = []
    expo_all = []

    for run in runs:
        # load
        if mode == "edges":
            exposure = pd.read_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_edges_{shuffle_type}_{run}.pkl")
            cns.a_ij = pd.read_pickle(f"{path}/a_ij_shuffled_time_{shuffle_type}_{run}.pkl")
        elif mode == "traits":
            exposure = pd.read_pickle(f"{path}/exposure_fitness_7_1_shuffled_time_traits_{shuffle_type}_{run}.pkl")
            cns.op_nodes = pd.read_pickle(f"{path}/op_nodes_shuffled_time_{shuffle_type}_{run}.pkl")
        if analysis == "expo_nmb":
            exposure.rename(columns={"exposure":"exposure_old", "n_influencer_summed":"exposure"},inplace=True)
        #restrict
        exposure.reset_index(inplace=True)
        exposure = exposure.loc[(exposure.time >= pd.to_datetime(start)) & (exposure.time <= pd.to_datetime(end))]
        exposure.set_index(["node_id","time"], inplace=True)
        # degree filter
        degree = exposure.groupby("node_id").n_nbs.mean().to_frame("avg").reset_index()
        exposure = exposure.loc[degree.loc[degree.avg >= 4,"node_id"]]
        exposure = exposure[exposure.n_nbs_mean > 1/7]
        #calculate
        data, expo_agg = cns_glasses.opinion_change_per_exposure(exposure, opinion_type, opinion_change_time = 1)
        data_all += [data]
        expo_all += [expo_agg]
    return pd.concat(data_all), pd.concat(expo_all), cns_glasses

# Surrogate 5
data_edges, expo_all_edges, cns_glasses_edges = load_data(runs=range(30), mode="edges", analysis = "expo_nmb", path= "tmp/ShuffledV4")
data_edges.to_pickle("tmp/final/surrogate5.pkl")

# Surrogate 1
data_traits, expo_all_traits, cns_glasses_traits = load_data(runs=range(30), mode="traits", analysis = "expo_nmb", path= "tmp/ShuffledV4")
data_traits.to_pickle("tmp/final/surrogate1.pkl")

# Surrogate 2
data_traits_pn, expo_all_traits_pn, cns_glasses_traits_pn = load_data(runs=range(30), mode="traits", analysis = "expo_nmb", path= "tmp/ShuffledV4", shuffle_type="complete_per_node")
data_traits_pn.to_pickle("tmp/final/surrogate2.pkl")