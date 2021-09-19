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
analysis = 'expo_frac'
opinion_type = "op_fitness"
binning = True
n_bins = 15
save_plots = False
show_plot = True

# define surrogate loader
def load_data(runs, analysis = "expo_frac", path= "tmp/ShuffledV4"):
    cns = CNSWorld()
    cns.load_world(opinions = ['fitness'], read_cached = True, stop=False, write_pickle = False, continous_op=False)
    cns_glasses = Glasses(cns)
    cns_glasses.output_folder = ""
    cns.d_ij = None
    
    start = "2014-02-01"
    end = "2014-04-30"
    cns.time = cns.time.loc[(cns.time.time >= start) & (cns.time.time <= end)]
    cns.op_nodes = cns.op_nodes.loc[(cns.op_nodes.time >= start) & (cns.op_nodes.time <= end)]
    
    data_all = []
    expo_all = []

    for run in runs:
        # load
        exposure = pd.read_pickle(f"{path}/exposure_fitness_7_1_shuffled_edges_cons_indiv_degree_{run}.pkl")
        cns.a_ij = pd.read_pickle(f"{path}/a_ij_shuffled_edges_cons_indiv_degree_{run}.pkl")
        cns.a_ij = cns.a_ij.loc[(cns.a_ij.time >= start) & (cns.a_ij.time <= end)]
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

# load data
data_edges, expo_edges, cns_glasses = load_data(range(5), analysis = "expo_nmb")
data_edges.to_pickle("tmp/final/surrogate6.pkl")

