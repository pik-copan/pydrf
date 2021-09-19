import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings


class World:

    DATA_PICKLE = 'data'
    REL_GRAPH_PICKLE = 'rel_graph'

    a_ij = pd.DataFrame() # relation network
    d_ij = pd.DataFrame() # like-mindedness network
    op_nodes = pd.DataFrame() # opinions of the nodes
    time = pd.DataFrame()
    data = pd.DataFrame()
    relation_graph = []
    like_mindedness_graph = []

    def __init__(self):
        return 0

    def get_data(self, read_cached = False):

        pickle_data_filename = self.DATA_PICKLE \
                                       + "_" + self.name \
                                       + "_" + '-'.join(self.d_ij.columns) \
                                       + ".pkl"
        if not self.data.empty:
            return self.data

        if read_cached :
            data_cached = False
            try:
                data = pd.read_pickle(self.PICKLE_PATH + pickle_data_filename)
                print("INFO: Reading data from cache")
                data_cached = True
            except FileNotFoundError:
                warnings.warn("No cached main dataframe found, create new dataframe.")
                data_cached = False

        if not (read_cached and data_cached):
            # import data
            a_ij = self.a_ij.drop_duplicates(['id_A','id_B','time'])
            data = self.d_ij.drop_duplicates(['id_A','id_B','time'])

            # create data set
            a_ij.set_index(['id_A','id_B','time'], inplace=True)
            data.set_index(['id_A','id_B','time'], inplace=True)
            data['edge'] = a_ij.edge
            #data = data.join(a_ij, how='outer')

            # fill in missing data
            default_edge = False
            data.edge.fillna(default_edge,inplace=True)

            # check for nulls
            if data.isnull().values.any():
                warnings.warn("Found a NULL in main data frame")

            self.data = data
            data.to_pickle(self.PICKLE_PATH + pickle_data_filename)

        return data

    def get_relation_graph(self, read_cached = False):

        pickle_rel_graph_filename = self.REL_GRAPH_PICKLE \
                                       + "_" + self.name \
                                       + "_" + '-'.join(self.d_ij.columns) \
                                       + ".pkl"

        if len(self.relation_graph) > 0:
            return self.relation_graph

        if read_cached :
            rel_graph_cached = False
            try:
                self.relation_graph = nx.read_gpickle(self.PICKLE_PATH + pickle_rel_graph_filename)
                rel_graph_cached = True
            except FileNotFoundError:
                warnings.warn("No cached relation graph found, create new relation graph.")
                rel_graph_cached = False

        if not (read_cached and rel_graph_cached):
            data = self.a_ij[self.a_ij.edge == 1]
            self.relation_graph = [(t, nx.from_pandas_edgelist(data.loc[data.time == t], source='id_A',target='id_B')) for t in self.time.time]

        return self.relation_graph
    
    def get_relation_graph_t(self, t):
        data = self.a_ij.loc[(self.a_ij.edge == 1) & (self.a_ij.time == t)]
        relation_graph_t = nx.from_pandas_edgelist(data, source='id_A',target='id_B')
        return relation_graph_t


    def get_like_mindedness_graph(self, opinion_type):
        for t in self.time.time:
            net = self.d_ij[(self.d_ij.time == t)]
            graph_t = nx.from_pandas_edgelist(net[net[opinion_type]==True], source='id_A',target='id_B')
            self.like_mindedness_graph.append((t,graph_t))
        return self.like_mindedness_graph


    def get_node_degree(self,network):
        # distinguish between relation network and like-mindedness network
        if network == 'relation':
            # check if network is loaded
            if self.a_ij.empty: raise ValueError('Relation network not yet loaded.')
            # return a dataframe with degree for each node
            return self._get_degree_t(self.a_ij, 'edge')
        elif network == 'like-mindedness':
            # check if network is loaded
            if self.d_ij.empty: raise ValueError('Like-mindedness network not yet loaded.')

            degree = pd.DataFrame() #init final dataframe
            # loop through all different opinions
            for op in self.d_ij.columns[self.d_ij.columns.str.startswith('op_')]:
                # calculate degree for given network considereing opinion 'op'
                degree_op = self._get_degree_t(self.d_ij, op)
                degree_op.rename(columns={'degree':'degree_'+op}, inplace=True)

                # join with final data frame
                if degree.empty:
                    degree = degree_op
                else:
                    degree.set_index(['node_id','time'], inplace=True)
                    degree_op.set_index(['node_id','time'], inplace=True)
                    degree = degree.join(degree_op, how='outer')
                    degree.reset_index(inplace=True)
            # for some nodes the degree is unknown -> set to 0
            degree.fillna(0,inplace=True)
            return degree
        else:
            raise ValueError('Network "'+ network +'" is unknown.')

    def get_common_friends(self):
        commons = pd.DataFrame() # final dataframe for common friends
        # loop through time steps
        for t in self.time.time :
            # create subgraph
            net = self.a_ij[(self.a_ij.time == t)]
            graph_t = nx.from_pandas_edgelist(net[net.edge==True], source='id_A',target='id_B')
            # calc common friends for each pair of nodes
            pair_of_nodes = zip(net.id_A,net.id_B)
            commons_t = list(map(lambda n: (n[0],n[1],self._number_common_neighbors(graph_t,n[0],n[1])),pair_of_nodes))
            # beautify dataframe and append to final dataframe
            commons_t = pd.DataFrame(commons_t)
            commons_t.columns = ['id_A', 'id_B', 'common_neighbors']
            commons_t['time'] = t
            commons = commons.append(commons_t)
        return commons

    def get_unlike_minded_friends_number(self, opinion, read_cached):

        # count unlike-minded friends
        data = self.get_data(read_cached = read_cached).reset_index()
        data = data[data.edge == True]
        data2 = data[data[opinion] == False]
        ulm_friendsA = data2.groupby(['id_A','time']).id_B.count() \
            .to_frame().reset_index().rename(columns={'id_A':'node_id','id_B':'ulm_friends_' + opinion})
        ulm_friendsB = data2.groupby(['id_B','time']).id_A.count() \
            .to_frame().reset_index().rename(columns={'id_B':'node_id','id_A':'ulm_friends_' + opinion})
        ulm_friends = pd.concat([ulm_friendsA, ulm_friendsB], sort=False)
        ulm_friends.drop_duplicates(['node_id','time'],inplace=True)
        ulm_friends.set_index(['node_id','time'],inplace=True)

        # pack into op_nodes data frame
        op_nodes = self.op_nodes.set_index(['node_id','time'])
        op_nodes['exposure'] = ulm_friends['ulm_friends_' + opinion]
        op_nodes['exposure'].fillna(0,inplace=True)
        op_nodes.reset_index(inplace=True)
        return op_nodes

    def get_unlike_minded_friends_fraction(self, opinion, read_cached):
        data = self.get_data(read_cached = read_cached).reset_index()
        data = data[data.edge == True]

        # count unlike-minded friends
        data2 = data[data[opinion] == False]
        ulm_friendsA = data2.groupby(['id_A','time']).id_B.count() \
            .to_frame().reset_index().rename(columns={'id_A':'node_id','id_B':'ulm_friends_' + opinion})
        ulm_friendsB = data2.groupby(['id_B','time']).id_A.count() \
            .to_frame().reset_index().rename(columns={'id_B':'node_id','id_A':'ulm_friends_' + opinion})
        ulm_friends = pd.concat([ulm_friendsA, ulm_friendsB], sort=False)
        ulm_friends.drop_duplicates(['node_id','time'],inplace=True)
        ulm_friends.set_index(['node_id','time'],inplace=True)

        # count like-minded friends
        data2 = data[data[opinion] == True]
        lm_friendsA = data2.groupby(['id_A','time']).id_B.count() \
            .to_frame().reset_index().rename(columns={'id_A':'node_id','id_B':'lm_friends_' + opinion})
        lm_friendsB = data2.groupby(['id_B','time']).id_A.count() \
            .to_frame().reset_index().rename(columns={'id_B':'node_id','id_A':'lm_friends_' + opinion})
        lm_friends = pd.concat([lm_friendsA, lm_friendsB], sort=False)
        lm_friends.drop_duplicates(['node_id','time'],inplace=True)
        lm_friends.set_index(['node_id','time'],inplace=True)


        total_friends = ulm_friends.join(lm_friends, how='outer').fillna(0)

        # calc fraction & return
        fraction = total_friends['ulm_friends_' + opinion] / (total_friends['ulm_friends_' + opinion] + total_friends['lm_friends_' + opinion])
        fraction = fraction.to_frame().rename(columns={0:'ulm_friends_' + opinion})

        # pack into op_nodes data frame
        op_nodes = self.op_nodes.set_index(['node_id','time'])
        op_nodes['exposure'] = fraction['ulm_friends_' + opinion]
        op_nodes['exposure'].fillna(0,inplace=True)
        op_nodes.reset_index(inplace=True)
        return op_nodes

    def get_exposure_to_opinion(self, opinion, access_values = False):

        exposed = self.op_nodes.groupby('time').apply(lambda p: \
                        self._compare_opinions(p, opinion, access_values, directed = True, consider_prev_op=False) )
        exposed.rename(columns={opinion:'exposed'},inplace=True)
        exposed.reset_index(inplace=True)
        exposed = exposed[exposed.id_A != exposed.id_B]

        # filter social relations
        friends = self.a_ij[self.a_ij.edge].set_index(['id_A','id_B','time'])
        exposed = friends.join(exposed.set_index(['id_A','id_B','time']))
        exposed.reset_index(inplace=True)

        # calc fraction of friends which are contagious
        exposed.rename(columns={'id_A':'node_id'},inplace=True)
        exposure = exposed.groupby(['node_id','time']).apply(lambda p: p[p.exposed].id_B.count() / p.id_B.count())

        # calc number of friends wich are contagious
        exposure_number = exposed[exposed.exposed].groupby(['node_id','time']).id_B.count()


        # pack into op_nodes data frame
        op_nodes = self.op_nodes.set_index(['node_id','time'])
        op_nodes['exposure'] = exposure
        op_nodes['exposure_number'] = exposure_number
        op_nodes['exposure'].fillna(0,inplace=True)
        op_nodes['exposure_number'].fillna(0,inplace=True)
        op_nodes.reset_index(inplace=True)

        return op_nodes

    def plot(self, time=None, show=True, save=False, suffix='', dir=''):
        raise ValueError("Plot function under construction.")
        if not time == None:
            net = self.a_ij[self.a_ij.time == time]
            op_nodes = self.op_nodes[self.op_nodes.time == time]
            graph_t = nx.from_pandas_edgelist(net[net.edge==True], source='id_A',target='id_B')
            nx.set_node_attributes(graph_t, pd.Series(op_nodes.op_politics, index=op_nodes.node_id).to_dict(), 'op_politics')
            #nx.draw(graph_t)
            nodes=graph_t.nodes()
            groups = set(nx.get_node_attributes(graph_t,'op_politics').values())
            mapping = dict(zip(groups,np.arange(0,len(groups))))
            colors = [mapping[graph_t.node[n]['op_politics']] for n in nodes]
            pos = nx.spring_layout(graph_t)
            ec = nx.draw_networkx_edges(graph_t, pos, alpha=0.2)
            nc = nx.draw_networkx_nodes(graph_t, pos, nodelist=nodes, node_color=colors, \
                            with_labels=False, node_size=100)
            if show: plt.show()
            if save: plt.savefig('Graph_'+suffix+'.png')
        else:
            for t in self.time.time:
                net = self.a_ij[self.a_ij.time == t]
                graph_t = nx.from_pandas_edgelist(net[net.edge==True], source='id_A',target='id_B')
                plt.clf()
                nx.draw(graph_t)
                if show: plt.show()
                if save: plt.savefig(dir + 'Graph_'+suffix+'_t='+t+'.png')


    def plot_fraction_of_connected_nodes(self):
        b = self.a_ij[self.a_ij.edge].groupby('time')['id_A','id_B'].apply(lambda b: len(list(b.id_A.unique()) + list(set(b.id_B.unique()) - set(b.id_A.unique())))/len(self.op_nodes.node_id.unique()))
        plt.plot(b)
        ticks = plt.xticks(rotation='vertical')
        plt.ylabel('Fraction of nodes with social relation')
        plt.show()

    def _number_common_neighbors(self, G, u, v):
        try:
            n = len(list(nx.common_neighbors(G,u,v)))
        except nx.NetworkXError:
            n = 0
        return n

    def _get_degree_t(self, net, edge_name):
        degree = pd.DataFrame()
        for t in self.time.time :
            graph_t = nx.from_pandas_edgelist(net[(net[edge_name]==True) & \
                                           (net.time == t)],\
                                            source='id_A',target='id_B')
            degree_t = pd.DataFrame(list(graph_t.degree),columns=['node_id','degree'])
            degree_t['time'] = t
            degree = degree.append(degree_t)
        return degree

    def _load_edges_from_opinions(self, op_nodes, time, access_values = False, consider_prev_op=True):
        d_ij = pd.DataFrame() #edges

        # loop through all opinion types
        for op in op_nodes.columns[op_nodes.columns.str.startswith('op_')]:

            d_ij_op = op_nodes.groupby(['time']).apply(lambda p: self._compare_opinions(p, op, access_values, consider_prev_op = consider_prev_op))
            d_ij_op.reset_index(inplace=True)


            # join all opinions together
            if d_ij.empty:
                d_ij = d_ij_op
            else:
                d_ij.set_index(['id_A','id_B','time'],inplace=True)
                d_ij_op.set_index(['id_A','id_B','time'],inplace=True)
                d_ij = d_ij.join(d_ij_op,how='outer')
                d_ij.reset_index(inplace=True)

        # Clean up:
        #   - likemindedness of same node: id_A=id_B
        #   - double counted likemindedness 1-2 & 2-1
        d_ij = d_ij[d_ij.id_A < d_ij.id_B]

        return d_ij

    def _load_edges_from_proximity(self, proxi , encounter_offset):

        proximity = proxi.copy()
        proximity['encounter'] = 1

        # substract 6 days, because pd.Grouper counts strange
        proximity.time = pd.to_datetime(proximity.time) - pd.to_timedelta(6, unit='d')
        # aggregate weekly
        proximity = proximity.groupby(['id_A','id_B', pd.Grouper(key='time', freq='W-MON')]).encounter \
                                .sum() \
                                .reset_index() \
                                .sort_values('time')

        meeting = proximity[['id_A','id_B','time','encounter']]
        meeting.time = meeting.time.astype('str')

        # merge tuples i.e. 1-2 and 2-1
        # take maximum of recorded encounters
        meeting_inv = meeting.rename(columns={'id_A' : 'id_B', 'id_B':'id_A'})
        meeting.set_index(['id_A','id_B','time'], inplace=True)
        meeting_inv.set_index(['id_A','id_B','time'], inplace=True)
        meeting = pd.concat([meeting, meeting_inv]).groupby(['id_A','id_B','time']).max()
        # do not count connections twice 1-2 & 2-1
        meeting.reset_index(inplace=True)
        meeting = meeting[meeting.id_A < meeting.id_B]
        meeting.set_index(['id_A','id_B','time'], inplace=True)

        # define if edge is established
        a_ij = meeting.encounter > encounter_offset
        a_ij = a_ij.to_frame()
        a_ij.rename(columns={'encounter':'edge'}, inplace=True)
        a_ij.reset_index(inplace=True)

        return a_ij

    def _compare_opinions(self, op_nodes, opinion_type, access_values = False, directed = False, consider_prev_op = True):

        # compare opinion & prev_opinion
        prob_of_op = "pop_" + opinion_type
        if access_values:
            # some how this fixes an key 0 error.
            opinion_matrix = np.array([op_nodes[opinion_type].values]*len(op_nodes))
            pop_matrix = np.array([op_nodes[prob_of_op].values]*len(op_nodes))
            if consider_prev_op:
                prev_opinion_matrix = np.array([op_nodes["prev_" + opinion_type].values]*len(op_nodes))
                prev_pop_matrix = 1-pop_matrix
        else:
            opinion_matrix = np.array([op_nodes[opinion_type]]*len(op_nodes))
            pop_matrix = np.array([op_nodes[prob_of_op]]*len(op_nodes))
            if consider_prev_op:
                prev_opinion_matrix = np.array([op_nodes["prev_" +opinion_type]]*len(op_nodes))
                prev_pop_matrix = 1-pop_matrix

        if opinion_type == 'op_music':
            opinion_matrix = opinion_matrix.astype('int')
            opinion_matrix_T = np.transpose(opinion_matrix,(1, 0, 2))
            if consider_prev_op:
                prev_opinion_matrix = prev_opinion_matrix.astype('int')
                prev_opinion_matrix_T = np.transpose(prev_opinion_matrix,(1, 0, 2))
        else:
            opinion_matrix_T = np.transpose(opinion_matrix)
            if consider_prev_op:
                prev_opinion_matrix_T = np.transpose(prev_opinion_matrix)

        pop_matrix_T = np.transpose(pop_matrix)
        if consider_prev_op:
            prev_pop_matrix_T = np.transpose(prev_pop_matrix)


        index = op_nodes.rename(columns={'node_id':'id_B'})
        colls = op_nodes.rename(columns={'node_id':'id_A'})

        d_ij = self._op_are_equal(opinion_type, opinion_matrix, opinion_matrix_T, pop_matrix, pop_matrix_T, axis=2, directed=directed)
        d_ij = pd.DataFrame(d_ij, index=index.id_B, columns=colls.id_A) \
                            .unstack().to_frame()
        # make the data frame nice
        d_ij.rename(columns={0:opinion_type},inplace=True)


        if consider_prev_op:
            prev_d_ij = self._op_are_equal(opinion_type, prev_opinion_matrix, prev_opinion_matrix_T, prev_pop_matrix, prev_pop_matrix_T, axis=2, directed=directed)
            prev_d_ij = pd.DataFrame(prev_d_ij, index=index.id_B, columns=colls.id_A) \
                                .unstack().to_frame()

            # set lik mindedness as True if d_ij or pref_dij are True
            d_ij[0] = prev_d_ij[0]
            d_ij[opinion_type] = d_ij[opinion_type] | d_ij[0]
            d_ij.drop(columns=0,inplace=True)

        return d_ij

    def get_opinion_change(self, op_nodes, opinion_type, shift = -1):
        op_nodes = op_nodes.copy()

        op_nodes['other_opinion'] = op_nodes.groupby(['node_id'])[opinion_type].shift(shift)
        op_nodes.dropna(inplace=True)
        
        prob_of_op = "pop_" + opinion_type
        no_change = self._op_are_equal(opinion_type, op_nodes[opinion_type], op_nodes['other_opinion'], op_nodes[prob_of_op], op_nodes[prob_of_op], axis=1)
        op_nodes['op_change'] = (no_change == False)

        # compares opinions considering the opinion type
        # if opinion_type == 'op_politics':
        #     op_change = op_nodes[opinion_type] != op_nodes['prev_opinion']
        # elif opinion_type == 'op_healthy_diet':
        #     op_change = op_nodes[opinion_type] != op_nodes['prev_opinion']
        # elif opinion_type == 'op_music':
        #     dist = np.linalg.norm((np.array([op_nodes[opinion_type]])-np.array([op_nodes['prev_opinion']]))[0,:,:].astype('int'),axis=1)
        #     op_change = dist >= 4
        # elif opinion_type == 'op_synthetic':
        #     op_change = op_nodes[opinion_type] != op_nodes['prev_opinion']
        # elif opinion_type == 'op_smoking':
        #     op_change = op_nodes[opinion_type] != op_nodes['prev_opinion']
        # elif opinion_type == 'op_physical':
        #     op_change = abs(op_nodes[opinion_type] - op_nodes['prev_opinion']) > 1
        # else:
        #     raise ValueError('The opinion type "' + opinion_type + '" is unknown.')
        # op_nodes['op_change'] = op_change

        return op_nodes


    def _op_are_equal(self, opinion_type, op1, op2, pop1, pop2, pop_threshold = 0.2, axis = None, directed = False):

        if not opinion_type in ['op_politics', 'op_healthy_diet', 'op_music','op_synthetic','op_smoking','op_physical', 'op_fitness', 'op_flying']:
            raise ValueError('The opinion type "' + opinion_type + '" is unknown.')

        # compare for prob of opinion
        if directed:
            pursued = pop2-pop1 > 0
        else:
            pursued = abs(pop1-pop2) < pop_threshold
            main_opinion = pop1 >= 0.5
            pursued = pursued & main_opinion

        # compares opinions considering the opinion type
        if opinion_type == 'op_music':
            if axis == 2:
                dist = np.linalg.norm(op1-op2, axis=axis)
            elif axis == 1:
                dist = np.linalg.norm((np.array([op1])-np.array([op2]))[0,:,:].astype('int'), axis=axis)
            else:
                raise ValueError('Axis "' + axis + '" is unknown.')
            are_equal = (dist < 4) & pursued
        if (opinion_type == "op_physical"):
            are_equal = abs(op1 - op2) <= 1
            are_equal = are_equal & pursued
        else:
            are_equal = (op1 == op2)
            are_equal = are_equal & pursued

        return are_equal
