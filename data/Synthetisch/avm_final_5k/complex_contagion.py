"""
A very simple form of complex contagion where at not one but two randomly drawn
neighbors must have the same opinion for a change to take place.

--> This model is not adaptive!
"""
from itertools import combinations
import numpy as np
import networkx as nx
from collections import Counter
import os
import shutil


class ComplexContagion():


    def __init__(self, N=100, M=250, G=8, noise_level=0):
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
        """
        # Set up the network, i.e., neighborhood structure
        possible_edges = np.array(list(combinations(range(N), 2)))
        edges = possible_edges[np.random.choice(len(possible_edges), M, replace=False)]

        graph = nx.Graph()
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
        graph = self._graph
        self._steps += 1

        # Pick one node at random
        active_node = np.random.choice(graph.nodes())
        active_opinion = graph.nodes[active_node]['opinion']

        if graph.degree(active_node) < 2:
            return

        # Pich one neighbor at random
        nbs = graph[active_node]
        nbr1, nbr2 = np.random.choice(list(nbs), 2, replace=False)

        if graph.nodes[nbr1]['opinion'] == graph.nodes[nbr2]['opinion']:
            graph.nodes[active_node]['opinion'] = graph.nodes[nbr1]['opinion']
            
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
