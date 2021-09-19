"""
Implementation of the adaptive voter model as presented in

Holme, Petter, and M. E. J. Newman. “Nonequilibrium Phase Transition in the
Coevolution of Networks and Opinions.” Physical Review E 74, no. 5 (November
10, 2006): 056108. https://doi.org/10.1103/PhysRevE.74.056108.
"""
from itertools import combinations
import numpy as np
import networkx as nx
from collections import Counter
import os
import shutil


class AVM():
    """Implementation of the adaptive voter model."""

    def __init__(self, N=100, M=250, G=8, phi=0, noise_level = 0):
        """
        Initialize a random network topology with the specified parameters.

        Parameters
        ----------

        N : int
            The number of nodes in the network.
        M : int
            The number of edges in the network.
        G : int
            The number of possible opinions.
        phi : float
            The rewiring probability.
        """
        # Set up the network, i.e., neighborhood structure
        possible_edges = np.array(list(combinations(range(N), 2)))
        edges = possible_edges[np.random.choice(len(possible_edges), M, replace=False)]

        graph = nx.Graph(phi=phi)
        graph.add_nodes_from(range(N))
        graph.add_edges_from(edges)

        opinions = {i: np.random.randint(G) for i in range(N)}
        nx.set_node_attributes(graph, opinions, 'opinion')

        self._graph = graph
        self._steps = 0
        self.noise_level = noise_level
        self.G = G


    def current_number_of_opinions(self):
        """The number of opinions still present in the network.

        Returns
        -------
        int
            The number of remaining opinions.
        """
        opinions = [data for node, data in self._graph.nodes(data='opinion')]
        return len(set(opinions))


    def is_consensus(self):
        """Check if the model has converged into its consensus state.

        Consensus is reached if only one opinion is left or the network
        fragments into disconnected components with one unique opinion each.

        Returns
        -------
        bool
            True if consensus is reached, False otherwise
        """
        graph = self._graph

        for i, j in graph.edges():
            if graph.nodes[i]['opinion'] != graph.nodes[j]['opinion']:
                return False

        return True


    def _one_update(self):
        """
        Perform one update for one randomly selected pair of nodes i and j.

        With probability phi the link between i and j is removed if their
        opinions differ and a new link is established between i and a
        previously disconnected node k such that i and k have the same opinion.

        With probability 1-phi i copies the opinion of j.
        """
        graph = self._graph
        self._steps += 1

        # Pick one node at random
        active_node = np.random.choice(graph.nodes())
        active_opinion = graph.nodes[active_node]['opinion']

        if not graph.degree(active_node):
            return

        # Pick one neighbor at random
        nbs = graph[active_node]
        random_nbr = np.random.choice(list(nbs))

        # Rewire the network structure with probability 'phi'
        if np.random.random() < graph.graph["phi"]:

            # Get the set of unconnected neighbors with same opinion
            new_neighbor = []
            for node, data in graph.nodes(data='opinion'):
                if (data == active_opinion) and (node not in nbs) and (node != active_node):
                    new_neighbor.append(node)

            if not new_neighbor:
                return

            # Select a new neighbor with same opinion at random and update
            # network
            new_neighbor = np.random.choice(new_neighbor)

            graph.remove_edge(active_node, random_nbr)
            graph.add_edge(active_node, new_neighbor)

        # Update opinions with probability 1-phi
        else:
            graph.nodes[active_node]['opinion'] = graph.nodes[random_nbr]['opinion']

        # change opinion at random
        if np.random.random() < self.noise_level:
            graph.nodes[active_node]['opinion'] = np.random.randint(self.G)


    def number_of_cross_links(self):
        graph = self._graph
        count = 0
        for i, j in graph.edges():
            if graph.nodes[i]['opinion'] != graph.nodes[j]['opinion']:
                count += 1

        return count


    def run(self, steps=1000):
        """
        Integrate the model forward.

        Parameters
        ----------
        steps : int
            The number of integration steps.
        """
        if not steps:
            steps = self.number_of_cross_links()
            print("#steps not specified. Intelligently using ", steps, "steps")

        for _ in range(steps):
            self._one_update()


    def current_steps(self):
        """
        The current number of total integration steps.

        Returns
        -------
        int
            The current number of total integration steps.
        """
        return self._steps


    def cluster_size_distribution(self):
        """
        The frequency of cluster sizes.

        Returns
        -------
        dict
            Keys are the size of a cluster, corresponding values give the
            frequency of that cluster size.
        """
        opinions = [data for _, data in self._graph.nodes(data='opinion')]
        frequency = Counter(opinions)
        size_distribution = Counter(frequency.values())
        return dict(size_distribution)


    def get_opinions(self):
        return [self._graph.nodes[n]['opinion'] for n in self._graph.nodes()]

    
    def get_edgelist(self):
        return self._graph.edges()


def main():

    filename = "adaptive_voter_model_phi0"
    
    if not os.path.exists("output/"+filename):
        os.makedirs("output/"+filename)

    # The same settings as in the Paper
    N = int(3200/16)
    M = int(6400/16)
    gamma = 10
    G = int(N / gamma)

    if filename == "adaptive_voter_model_phi0":
        phi = 0
        steps = N
    elif filename == "secretmodel2":
        phi = 1
        steps = 250
    elif filename == "secretmodel3":
        phi = 0.25
        steps = 2500
    elif filename == "secretmodel4":
        phi = 0.75
        steps = 500

    foldername = "output/" + filename + "/"
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)
    
    filename = foldername + filename

    print(phi, steps)
    model = AVM(N=N, M=M, G=G, phi=phi)

    counter = 0
    np.savetxt(filename+'_opinions_t{0}.txt'.format(counter), model.get_opinions(), fmt="%i")
    np.savetxt(filename+'_edges_t{0}.txt'.format(counter), model.get_edgelist(), fmt="%i")

    while not model.is_consensus():
        counter += 1
        model.run(steps=steps)
        print(counter, model.current_steps(), model.current_number_of_opinions())
        np.savetxt(filename+'_opinions_t{0}.txt'.format(counter), model.get_opinions(), fmt="%i")
        np.savetxt(filename+'_edges_t{0}.txt'.format(counter), model.get_edgelist(), fmt="%i")

    print(Counter(model.get_opinions()))
    #print(model.cluster_size_distribution())


if __name__ == "__main__":
    main()
