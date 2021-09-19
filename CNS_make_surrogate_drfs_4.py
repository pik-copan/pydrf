from sensible_raw.loaders import loader
from world_viewer.cns_world import CNSWorld
from world_viewer.glasses import Glasses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LogNorm
from sklearn.utils import shuffle
import matplotlib.dates as mdates
from matplotlib.figure import figaspect

# load tools
cns = CNSWorld()
cns.load_world(opinions = ['fitness'], read_cached = True, stop=False, write_pickle = False, continous_op=False)
cns_glasses = Glasses(cns)
cns.d_ij = None

# set parameters for analysis
analysis = 'expo_frac'
opinion_type = "op_fitness"
binning = True
n_bins = 15
save_plots = False
show_plot = True
start = "2014-02-01"
end = "2014-04-30"
path = "tmp/Shuffled"

# function for loading surrogates
def load_data(runs, analysis, exposure_time):
    data_all = []
    expo_agg_all = []
    for run in runs:
        # read exposure
        exposure = pd.read_pickle(f"{path}/exposure_fitness_{exposure_time}_1_shuffled_op_fitness_{run}.pkl")
        # read nodes
        cns.op_nodes = pd.read_pickle(f"{path}/op_nodes_fitness_shuffled_op_fitness_{run}.pkl")
        
        if analysis == "expo_nmb":
                exposure.rename(columns={"exposure":"exposure_old", "n_influencer_summed":"exposure"},inplace=True)

        # degree filter
        degree = exposure.groupby("node_id").n_nbs.mean().to_frame("avg").reset_index()
        exposure = exposure.loc[degree.loc[degree.avg >= 4,"node_id"]]
        exposure = exposure[exposure.n_nbs_mean > 1/7]

        # time filter
        exposure.reset_index(inplace=True)
        exposure = exposure.loc[(exposure.time >= pd.to_datetime(start)) & (exposure.time <= pd.to_datetime(end))]
        exposure.set_index(["node_id","time"], inplace=True)

        # do calcs
        data, expo_agg = cns_glasses.opinion_change_per_exposure(exposure, opinion_type, opinion_change_time = 1)
        data_all += [data]
        expo_agg_all += [expo_agg]

    return pd.concat(data_all), pd.concat(expo_agg_all)

# load surrogate
data_trait, exposure_trait = load_data(range(30), analysis="expo_nmb", exposure_time=7)
data_trait.to_pickle("tmp/final/surrogate4.pkl")
