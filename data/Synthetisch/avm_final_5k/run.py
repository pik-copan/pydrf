from complex_contagion import ComplexContagion
from adaptive_voter_model import AVM
import os
import numpy as np
import shutil
from collections import Counter
import random
import sys
import matplotlib.pyplot as plt



def main():

    # mode can either be complex or simple
    mode = sys.argv[2]
    phi = float(sys.argv[1])

    
    run_nmb = int(sys.argv[3])

    random.seed(run_nmb)
    np.random.seed(run_nmb)
    
    output_folder = "./"
    N = 851
    M = 5724
    max_time = 89

    if phi == 0.6:
        step_factor = 0.105 # number of iterations per time step, as fraction of N.
    elif phi == 0.0:
        step_factor = 0.105

    # Number of opinions
    #G = 20
    G = 2

    tc = True
    noise_level = 0#float(sys.argv[1])
    
    if tc:
        steps = int(step_factor * N/(1-phi))
    else:
        steps = N

    if mode == "complex":
        filename = f"complex_contagion_phi{phi}_nopinions{G}_run{run_nmb}"
        model = ComplexContagion(N=N, M=M, G=G, noise_level= noise_level)
    elif mode == "simple":
        if (step_factor != None):
            filename = f"adaptive_voter_model_phi{phi}_nopinions{G}_steps{steps}_run{run_nmb}"
        else:
            filename = f"adaptive_voter_model_phi{phi}_nopinions{G}_run{run_nmb}"
        model = AVM(N=N, M=M, G=G, phi=phi, noise_level= noise_level)
    if tc:
        filename = filename + "_tc"

    foldername = output_folder + filename + "/"

    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)
    
    filename = foldername + filename

    counter = 0
    np.savetxt(filename+f'_opinions_t{counter}.txt', model.get_opinions(), fmt="%i")
    np.savetxt(filename+f'_edges_t{counter}.txt', model.get_edgelist(), fmt="%i")

    #plt.axis()
    while (not model.is_consensus()) and (counter < max_time):
        counter += 1
        model.run(steps=steps)
        print(counter, model.current_steps(),
              model.current_number_of_opinions(),
              model.cluster_size_distribution())
        #print(counter, model.number_of_cross_links())
        #plt.plot(counter, model.number_of_cross_links(), marker="x")
        #plt.pause(0.05)
        np.savetxt(filename+f'_opinions_t{counter}.txt', model.get_opinions(), fmt="%i")
        np.savetxt(filename+f'_edges_t{counter}.txt', model.get_edgelist(), fmt="%i")

    #plt.show()
    print(Counter(model.get_opinions()))


if __name__ == "__main__":
    main()
