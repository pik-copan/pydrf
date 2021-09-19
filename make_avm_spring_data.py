import sys
# from world_viewer.mit_world import MITWorld
from world_viewer.synthetic_world import SyntheticWorld
from world_viewer.glasses import Glasses
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

run=1
phi=0.0

syn = SyntheticWorld(path="data/Synthetisch/avm_final_5k", run=run, number_of_nodes=851)
syn.load_world(phi=phi, cc=False, n_op=2 ,read_cached=False, tc = True, steps = 89)
syn_glasses = Glasses(syn)

opinion_type = "op_synthetic"

exposure = pd.read_pickle(f"tmp/final/avm_final/exposure_phi{phi}_run{run}_5k.pkl")
exposure.rename(columns={"exposure":"exposure_old", "n_influencer_summed":"exposure"},inplace=True)
data, expo_agg = syn_glasses.opinion_change_per_exposure(exposure, opinion_type, opinion_change_time = 1)
data.to_pickle("tmp/final/spring_data_phi0.0.pkl")