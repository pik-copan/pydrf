import sys
# from world_viewer.mit_world import MITWorld
from world_viewer.synthetic_world import SyntheticWorld
from world_viewer.glasses import Glasses
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

phi =  float(sys.argv[1])
#phi = 0.8
cc = bool(int(sys.argv[2]))
#cc = False
n_ops = int(sys.argv[3])
#n_ops = int(sys.argv[1])
#survey_size = 5
tc = bool(int(sys.argv[4]))
#tc = bool(int(sys.argv[2]))
steps = int(sys.argv[5])
run = int(sys.argv[6])


noise_level = 0.0
output_folder = "tmp/final/avm_final/"

syn = SyntheticWorld(path="data/Synthetisch/avm_final_5k", run=run, number_of_nodes=851)
syn.load_world(phi=phi, cc=cc, n_op=n_ops, steps=steps, read_cached=False, tc=tc)
op_changes = syn.get_opinion_changes_per_timestep()
plt.hist(op_changes, label="Opinion changes per day")
plt.axvline(op_changes.mean(), label="Mean", color="red")
plt.savefig("tmp/final/avm_final/OpChPerDay_phi"+str(phi)+"_steps"+str(steps)+"_run"+str(run)+".png")
syn_glasses = Glasses(syn)
exposure = syn_glasses.calc_exposure("expo_frac", "op_synthetic", exposure_time=7)
exposure.to_pickle(output_folder + f"exposure_phi{phi}_run{run}_5k.pkl")
