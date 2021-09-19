from sensible_raw.loaders import loader
from world_viewer.cns_world import CNSWorld
from world_viewer.glasses import Glasses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LogNorm
from sklearn.utils import shuffle

folder = "tmp/Shuffled/"

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



# shuffle trait
for run in range(15,30):
    cns.op_nodes.set_index("time",inplace=True)
    cns.op_nodes.sort_index(inplace=True)
    cns.op_nodes["op_fitness"] = cns.op_nodes.groupby("time").op_fitness.apply(shuffle)\
                     .reset_index().set_index("time").op_fitness
    cns.op_nodes.reset_index(inplace=True)
    cns.op_nodes.to_pickle(folder + f"op_nodes_fitness_shuffled_op_fitness_{run}.pkl")
    
    exposure = cns_glasses.calc_exposure("expo_frac", "op_fitness", exposure_time = 7)
    exposure.to_pickle(folder + f"exposure_fitness_7_1_shuffled_op_fitness_{run}.pkl")
