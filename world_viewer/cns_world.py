from world_viewer.world import World
import pandas as pd
import numpy as np
import warnings
# from sensible_raw.loaders import loader
import json
from math import ceil
import os
os.environ['R_HOME'] = '/home/lochnerpik@gmail.com/master/lib/R'


class CNSWorld(World):

    PICKLE_PATH = './pickle/' # path for cached data
    RELATION_NET_PICKLE = 'CNS_relation_net'
    OPINIONS_PICKLE = 'CNS_opinions'
    LIKE_MINDEDNESS_PICKLE = 'CNS_like_mindedness'
    
    CNS_TIME_BEGIN = pd.Timestamp(pd.datetime(2013, 9, 2)) # first timestamp 
    CNS_TIME_END = pd.Timestamp(pd.datetime(2014, 12, 31)) # last timestamp
    
    
    sigma = pd.to_timedelta(3, unit='d').total_seconds()
    two_sigma_sqr = 2* sigma * sigma

    def __init__(self, path='', start=pd.datetime(2013, 9, 2), end=pd.datetime(2014, 12, 31)):
        super().__init__()
        self.path = path
        self.CNS_TIME_BEGIN = start
        self.CNS_TIME_END = end

    def load_world(self, opinions = ['smoking'], relation_agg = 2, read_cached = False, stop=False, write_pickle = True, continous_op = False):
        self.name = "CNS" + '-'.join(opinions)
        self.type = "CNS"
        
        if continous_op:
            warnings.warn("No comparison of continous opinions implementet yet!")
        
        pickle_relation_net_filename = self.RELATION_NET_PICKLE \
                                       + "_" + str(relation_agg) \
                                       + ".pkl"
        pickle_opinions_filename = self.OPINIONS_PICKLE \
                                       + "_" + '-'.join(opinions) \
                                       + ".pkl"
        pickle_like_mindedness_filename = self.LIKE_MINDEDNESS_PICKLE \
                                       + "_" + '-'.join(opinions) \
                                       + ".pkl"
        
        ## 0. Load time
        #time = pd.DataFrame(pd.date_range(self.CNS_TIME_BEGIN, self.CNS_TIME_END, freq='W-MON'),columns=['time'])
        time = pd.DataFrame(pd.date_range(self.CNS_TIME_BEGIN, self.CNS_TIME_END, freq='d'),columns=['time'])
        self.time = time
            
        ## 1. Load opinions
        
        if read_cached:
            opinions_cached = False
            try:
                op_nodes = pd.read_pickle(self.PICKLE_PATH + pickle_opinions_filename)
                opinions_cached = True
            except FileNotFoundError:
                warnings.warn("No cached opinions found, read opinions from file.")
                opinions_cached = False

        if not (read_cached and opinions_cached):
            op_nodes = pd.DataFrame() # general opinion dataframe
            
            if len(list(set(opinions) & set(["smoking","physical"]))) > 0:
                op_data = pd.DataFrame() # df for loaded data
                # load data
                for survey in np.arange(1,4):
                    print('Load survey ' + str(survey))
                    data_s = loader.load_data("questionnaires", "survey_"+str(survey), as_dataframe=True) 
                    data_s = data_s[data_s.user < 1000] #clean strange users
                    op_time = self._get_op_time(survey)
                    data_s = data_s.set_index('user').join(op_time)
                    data_s = data_s[data_s.time.astype('int') > 10]
                    data_s[data_s.time < self.CNS_TIME_BEGIN] = self.CNS_TIME_BEGIN
                    data_s[data_s.time > self.CNS_TIME_END] = self.CNS_TIME_END
                    data_s['survey'] = survey
                    data_s.reset_index(inplace=True)
                    op_data = pd.concat([op_data,data_s],sort=False)

                #possibilitie that users filled out more than one questionaires in one week
                op_data.drop_duplicates(['user','time','variable_name'], keep='last', inplace=True) 
            
            # process opinions
            for opinion in opinions:

                # load smoking opinions
                if opinion == "smoking":
                    print("Process opinion data for variable: smoking")
                    opinion = "op_" + opinion
                    smoking = op_data[op_data.variable_name == b'smoke_freq'].copy()
                    smoking[opinion] = (smoking.response != b'nej_jeg_har_aldrig_r') \
                                        & (smoking.response != b'nej_men_jeg_har_rget')
                    smoking.reset_index(inplace=True)
                    smoking = smoking[['user', 'time', opinion, 'survey' ]]
                    smoking.rename(columns={'user':'node_id'},inplace=True)
                    smoking = self._add_time_to_op_nodes(smoking, time, opinion)
                    
                    # write into general dataframe
                    if op_nodes.empty:
                        op_nodes = smoking
                    else:
                        op_nodes = op_nodes.set_index(['node_id','time']).join(smoking.set_index(['node_id','time']), how='outer')
                        op_nodes.reset_index(inplace=True)
                        
                # load physical opinions
                elif opinion == "physical":
                    print("Process opinion data for variable: physical")
                    opinion = "op_" + opinion
                    physical = op_data[op_data.variable_name == b'physical_activity'].copy()
                    physical.response.replace(b'ingen',0,inplace=True)
                    physical.response.replace(b'ca__time_om_ugen',0,inplace=True)
                    physical.response.replace(b'ca_1-2_timer_om_ugen',1,inplace=True)
                    physical.response.replace(b'ca_3-4_timer_om_ugen',2,inplace=True)
                    physical.response.replace(b'ca_5-6_timer_om_ugen',3,inplace=True)
                    physical.response.replace(b'7_timer_om_ugen_elle',4,inplace=True)
                    physical.rename(columns={'response':opinion, 'user':'node_id'},inplace=True)
                    physical = physical[['node_id', 'time', opinion, 'survey' ]]
                    physical = self._add_time_to_op_nodes(physical, time, opinion)

                    # write into general dataframe
                    if op_nodes.empty:
                        op_nodes = physical
                    else:
                        op_nodes = op_nodes.set_index(['node_id','time','survey']) \
                                    .join(physical.set_index(['node_id','time','survey']), how='outer')
                        op_nodes.reset_index(inplace=True)
                        
                elif opinion == "fitness":
                    print("Process opinion data for variable: fitness")
                    opinion = "op_" + opinion
                    fitness = pd.read_pickle('data/op_fitness.pkl').reset_index()
                    fitness = fitness[['node_id','time','op_fitness_abs']]
                    fitness = fitness.rename(columns={"op_fitness_abs":"fitness"})
                    fitness["op_fitness"] = 0
                    fitness.sort_values(['node_id', 'time'], inplace=True)
                    fitness = fitness[fitness.time >= self.CNS_TIME_BEGIN]
                    fitness = fitness[fitness.time <= self.CNS_TIME_END]
                    fitness.set_index('node_id', inplace=True)

                    fitness.reset_index(inplace=True)
                    # discretize opinion
                    fitness.loc[fitness.fitness >= 1, "op_fitness"] = True
                    fitness.loc[fitness.fitness < 1, "op_fitness"] = False

                    # write into general dataframe
                    if op_nodes.empty:
                        op_nodes = fitness
                    else:
                        op_nodes = op_nodes.set_index(['node_id','time','survey']) \
                                    .join(fitness.set_index(['node_id','time','survey']), how='outer')
                        op_nodes.reset_index(inplace=True)

                else:
                    raise ValueError('The opinion "' + opinion + '" is unknown.')

            if write_pickle: op_nodes.to_pickle(self.PICKLE_PATH + pickle_opinions_filename)
            
        #save opinions as instance variable
        self.op_nodes = op_nodes
                
        if stop: return 0
            
       
        ## 3. Load relation network    
        
        relations = pd.read_pickle("data/relations.pkl")
        relations.reset_index(inplace=True)
        relations = relations[relations.time >= self.CNS_TIME_BEGIN]
        relations = relations[relations.time <= self.CNS_TIME_END]
        
        # take only nodes for which the opinion is known
        relations = relations[relations.id_A.isin(self.op_nodes.node_id)]
        relations = relations[relations.id_B.isin(self.op_nodes.node_id)]
        
        self.a_ij = relations[['id_A', 'id_B', 'time', 'edge']]

    
    def _get_op_time(self, survey):
        with open('user_scores'+str(survey)+'.json') as f:
                op_time = json.load(f)
        op_time = pd.DataFrame(op_time).loc['ts'].to_frame()
        op_time.index.name = 'user'
        op_time.reset_index(inplace=True)
        op_time.user = op_time.user.astype('int')
        op_time.set_index('user',inplace=True)
        op_time.rename(columns={'ts':'time'},inplace=True)
        op_time.time = pd.to_datetime(op_time.time, unit='s').dt.to_period('W').dt.to_timestamp()
        return op_time
    
    def load_edges_from_bluetooth2(self, proxi, time, verbose=True): #, threshold = None, verbose=True):
    
        proxi = proxi.copy()
        
        # take both directions id_A->id_B, id_B->id_A
        proxi_inv = proxi.rename(columns={'id_A':'id_B','id_B':'id_A'})
        proxi = pd.concat([proxi, proxi_inv], sort=False)
        proxi.drop_duplicates(['id_A','id_B','time'],inplace=True)
        # dont count edges twice
        proxi = proxi[proxi.id_A < proxi.id_B]

        proxi.time = proxi.time.dt.round('D')
        
        # count encounters per day
        proxi['encounter'] = 1
        proxi = proxi.groupby(['id_A','id_B','time']).encounter.sum().reset_index()
        
        print("before")
        print(proxi)
        #insert time steps with no recorded encounter
        proxi = proxi.groupby(['id_A','id_B'])[['time','encounter']] \
                .apply( lambda p: \
                       pd.DataFrame(p).set_index(['time']).join(time.set_index(['time']), how='outer') \
                      )
        proxi.reset_index(inplace=True)

        # fill unknown encounters with 0
        proxi.fillna(0,inplace=True)
        print("after")
        print(proxi)

        # weighted sum over a week
        proxi = proxi.groupby(['id_A','id_B'])['time','encounter'].apply(self._calc_interaction)
        
        proxi.reset_index(inplace=True)
        proxi.time = pd.to_datetime(proxi.time, unit='s')#.dt.to_period('W').dt.to_timestamp()
        #proxi = proxi.groupby(['id_A','id_B','time']).mean()
        
        self.meetings = proxi.reset_index()

        #determine edges
        #if threshold:
        #    proxi['edge'] = proxi.encounter > threshold   
        #    print("Use a_ij threshold: " + threshold)
        #else:
        #    proxi['edge'] = proxi.encounter > proxi.encounter.describe()['25%']

        return proxi.reset_index()


    def _calc_interaction(self,proxi_slice):
        proxi = proxi_slice.copy()
        proxi.time = proxi.time.astype('int')/1000000000.0 # to seconds
        time_matrix = np.array([proxi.time.values]*len(proxi.time))
        diff = time_matrix - time_matrix.transpose()
        matrix = np.exp(-(diff * diff)/self.two_sigma_sqr)
        filter_past = np.tril(np.ones_like(diff))
        matrix *= filter_past
        proxi.encounter =  np.dot(matrix, proxi.encounter)
        return proxi.set_index('time')
        
    def load_edges_from_bluetooth(self, proxi, encounter_offset, freq = 'weekly', time_format = 'ms'):

        proximity = proxi.copy()

        proximity['encounter'] = 1

        if freq == 'monthly':
            # convert time to datetime format
            proximity.time = pd.to_datetime(proximity.time, unit=time_format)
            # aggregate monthly
            proximity = proximity.groupby(['id_A','id_B', pd.Grouper(key='time', freq='M')]).encounter \
                                    .sum() \
                                    .reset_index() \
                                    .sort_values('time')
        elif freq == 'weekly':
            # substract 6 days, because pd.Grouper counts strange
            proximity.time = pd.to_datetime(proximity.time, unit=time_format) - pd.to_timedelta(6, unit='d')
            # aggregate weekly
            proximity = proximity.groupby(['id_A','id_B', pd.Grouper(key='time', freq='W-MON')]).encounter \
                                    .sum() \
                                    .reset_index() \
                                    .sort_values('time')
        else:
            raise ValueError('The frequency "' + freq + '" is unknown.')
            return -1

        meeting = proximity[['id_A','id_B','time','encounter']]
        meeting.time = meeting.time.astype('str')

        # merge tuples i.e. 1-2 and 2-1
        # take maximum of recorded encounters
        meeting_inv = meeting.rename(columns={'id_A' : 'id_B', 'id_B':'id_A'})
        meeting.set_index(['id_A','id_B','time'], inplace=True)
        meeting_inv.set_index(['id_A','id_B','time'], inplace=True)
        meeting = pd.concat([meeting, meeting_inv]).reset_index()
        meeting = meeting.groupby(['id_A','id_B','time']).max()
        # do not count connections twice 1-2 & 2-1
        meeting.reset_index(inplace=True)
        meeting = meeting[meeting.id_A < meeting.id_B]
        meeting.set_index(['id_A','id_B','time'], inplace=True)
        self. meetings = meeting


        # define if edge is established
        a_ij = meeting.encounter > encounter_offset
        a_ij = a_ij.to_frame()
        a_ij.rename(columns={'encounter':'edge'}, inplace=True)
        a_ij.reset_index(inplace=True)

        self.a_ij = a_ij

        return a_ij
    
    def _add_time_to_a_ij(self, a_ij, time):

        a_ij = a_ij.copy()

        # insert time into a_ij
        #a_ij = pd.merge_ordered(a_ij, time, left_by=['id_A','id_B'], how='outer')

        a_ij = a_ij.groupby(['id_A','id_B'])[['time','edge']] \
                .apply( lambda p: \
                       pd.DataFrame(p).set_index(['time']).join(time.set_index(['time']), how='outer') \
                      )
        a_ij.reset_index(inplace=True)

        # fill unknown edges with False = "no edge"
        a_ij.fillna(False,inplace=True)

        return a_ij
    
    def _add_time_to_op_nodes(self, nodes, time, opinion):

        nodes = nodes.copy()
        
        # create column for opinion at previous time step
        # at times where the survey was done, prev_opinion = opinion
        prev_opinion = 'prev_' + opinion
        survey = "survey"
        nodes[prev_opinion] = nodes[opinion]

        # insert missing time steps
        nodes = pd.merge_ordered(nodes, time, left_by='node_id', how='outer')
        
        # mark all data points between survey i and j with survey=j
        # the remaining dara points, after the last survey are survey=0
        nodes.survey = nodes.groupby('node_id').survey.fillna(method='bfill')
        nodes.survey = nodes.survey.fillna(0)
        
        # handle nan's in op_nodes
        nodes[opinion] = nodes.groupby('node_id')[opinion].bfill().ffill()
        nodes[prev_opinion] = nodes.groupby('node_id')[prev_opinion].ffill().bfill()

        # calc probability of opinion
        prob_of_op = 'pop_'+ opinion
        nodes[prob_of_op] = 0
        nodes[prob_of_op] = nodes.groupby(['node_id',survey])[prob_of_op] \
                                    .apply(lambda p: p + np.linspace(0,1,len(p)+1)[1:])
        nodes.loc[nodes[opinion] == nodes[prev_opinion],prob_of_op] = 1
        return nodes
    
    def _add_time_to_continous_op_nodes(self, nodes, time, opinion):

        nodes = nodes.copy()
        
        survey = "survey"

        # insert missing time steps
        nodes = pd.merge_ordered(nodes, time, left_by='node_id', how='outer')
        
        # mark all data points between survey i and j with survey=j
        # the remaining data points, after the last survey are survey=0
        nodes.survey = nodes.groupby('node_id').survey.fillna(method='bfill')
        nodes.survey = nodes.survey.fillna(0)
        
        prob_of_op = 'pop_'+ opinion
        nodes[prob_of_op] = 1
        
        nodes[opinion] = nodes[opinion].interpolate()
        nodes.dropna(inplace=True)
        
        return nodes

    def filter_dates(self, start_stop = False, christmas = True):
        
        # cut at start and end
        if start_stop:
            start = pd.Timestamp(pd.datetime(2013, 10, 8)) # first timestamp 
            stop = pd.Timestamp(pd.datetime(2014, 4, 22)) # last timestamp 

            self.a_ij = self.a_ij[self.a_ij.time >= start]
            self.a_ij = self.a_ij[self.a_ij.time <= stop]
            self.d_ij = self.d_ij[self.d_ij.time >= start]
            self.d_ij = self.d_ij[self.d_ij.time <= stop]
            self.op_nodes = self.op_nodes[self.op_nodes.time >= start]
            self.op_nodes = self.op_nodes[self.op_nodes.time <= stop]

        # pad christmas days
        if christmas:
            self.a_ij.drop(self.a_ij[self.a_ij.time == '2013-12-23'].index,inplace=True)
            self.a_ij.drop(self.a_ij[self.a_ij.time == '2013-12-30'].index,inplace=True)

            pad = self.a_ij[self.a_ij.time == '2013-12-16'].copy()
            pad.time = pd.Timestamp(pd.datetime(2013, 12, 23))
            self.a_ij = self.a_ij.append(pad)
            pad = self.a_ij[self.a_ij.time == '2014-01-06'].copy()
            pad.time = pd.Timestamp(pd.datetime(2013, 12, 30))
            self.a_ij = self.a_ij.append(pad)

            self.a_ij.sort_values(["time"],inplace=True)



        
