from world_viewer.world import World
import pandas as pd
import numpy as np
import warnings
import os
from math import floor
from itertools import product


class SyntheticWorld(World):

    TIME_POINTS = 20 #number of timesteps in the data
    TIME_STEP = 1
    NUMBER_NODES = 200 #number of nodes/people in the world
    PICKLE_PATH = './pickle/' # path for cached data
    RELATION_NET_PICKLE = 'SYN_relation_net'
    OPINIONS_PICKLE = 'SYN_opinions'
    LIKE_MINDEDNESS_PICKLE = 'SYN_like_mindedness'

    def __init__(self, path='data/Synthetisch', run = None, number_of_nodes = 200):
        super().__init__()
        self.path = path
        self.run = run
        self.NUMBER_NODES = number_of_nodes
        self.name = "SYN"


    def get_opinion_changes_per_timestep(self):
        """designed by Niklas for the AVM model, no idea how it does for others..."""

        # sort df so it lists all timesteps for node0, then all timesteps for node1 and so forth
        op_nodes = self.op_nodes.sort_values(by=["node_id", "time"]).copy()
        # Add column with the last timestep's opinion
        op_nodes["prev_step"] = op_nodes["op_synthetic"].shift(1)
        # remove all rows for the first timestep, since it doesn't have a previous timestep
        op_nodes = op_nodes[op_nodes["time"] != op_nodes["time"][0]]
        # add column with bool for opinion change since last time step
        op_nodes["op_change"] = op_nodes["op_synthetic"] != op_nodes["prev_step"]
        op_changes = op_nodes.groupby("time")["op_change"].sum().values
        print("Mean opinion changes per timestep: " + str(op_changes.mean()))
        return op_changes

    def load_world(self, phi = 0, cc = False, n_op = 2, steps=None, read_cached = False, tc=True, load_full_connection_df = False):
        ''' load an process adaptive voter model data '''
        # name to lable files
        self.name = "SYN_phi=" + str(phi) + "_CC=" + str(cc) + "_N=" + str(n_op) + "_TC=" + str(tc)

        if self.run != None:
            self.name += "_run=" + str(self.run)
        self.cc = cc
        self.type = "SYN"

        # names of cache files
        pickle_relation_net_filename = self.RELATION_NET_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_opinions_filename = self.OPINIONS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_like_mindedness_filename = self.LIKE_MINDEDNESS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        # filename of model data
        if cc:
            self.file_name = "complex_contagion_phi0.0_nopinions" + str(n_op)
        else:
            self.file_name = "adaptive_voter_model_phi" + str(phi) + "_nopinions" + str(n_op)

        if steps != None:
            self.file_name = self.file_name + "_steps" + str(steps)

        if self.run != None: self.file_name += "_run" + str(self.run)

        if tc: #time correction
            self.file_name = self.file_name + "_tc"

        print("Load World: " + self.name + " from " + self.file_name)

        # get number of time steps
        if not os.path.isdir(self.path +"/"+ self.file_name +"/"):
            raise ValueError('No model data found for given parameters at: ' + self.path +"/"+ self.file_name +"/")

        path, dirs, files = next(os.walk(self.path +"/"+ self.file_name +"/"))
        file_count = len(files)
        self.TIME_POINTS = floor(file_count/2)

        print("Numer of time steps is: " + str(self.TIME_POINTS))

        #### 0. define dummy time ####
        ##############################
        self.time = pd.date_range(start='1/1/2018', periods=self.TIME_POINTS/self.TIME_STEP, freq='W-Mon')
        self.time = pd.DataFrame(self.time.astype('str'),columns=['time'])

        #### 1. read relation network edges ####
        ########################################

        # try to read cache
        if read_cached:
            relations_cached = False
            try:
                self.a_ij = pd.read_pickle(self.PICKLE_PATH + pickle_relation_net_filename)
                print("INFO: Read relations from cache.")
                relations_cached = True
            except FileNotFoundError:
                warnings.warn("No cached relation network found, read relations from file.")
                relations_cached = False


        # if not succeded with reading cache
        if not (read_cached and relations_cached):
            print("read edges")

            nodes = range(self.NUMBER_NODES)

            if load_full_connection_df:
                self.a_ij = pd.DataFrame(list(product(nodes, nodes, self.time.time)), columns=['id_A', 'id_B', 'time'])
                self.a_ij = self.a_ij[self.a_ij.id_A != self.a_ij.id_B]
                self.a_ij['edge'] = False
                # a_ij = pd.DataFrame() # data frame for established edges
                a_ij_list = []
                for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                    edges_t = pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_edges_t"+str(t)+".txt",header=None,sep=' ',names=['id_A','id_B'])
                    edges_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                    edges_t['edge'] = True
                    a_ij_list.append(edges_t)
                a_ij = pd.concat(a_ij_list)
                self.a_ij.set_index(['id_A','id_B','time'], inplace=True)
                a_ij.set_index(['id_A','id_B','time'], inplace=True)
                self.a_ij.update(a_ij)
                self.a_ij.reset_index(inplace=True)
                self.a_ij.to_pickle(self.PICKLE_PATH + pickle_relation_net_filename)
            else:
                self.a_ij = pd.DataFrame()
                for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                    edges_t = pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_edges_t"+str(t)+".txt",header=None,sep=' ',names=['id_A','id_B'])
                    edges_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                    edges_t['edge'] = True
                    self.a_ij = self.a_ij.append(edges_t)
                self.a_ij.to_pickle(self.PICKLE_PATH + pickle_relation_net_filename)

        #### 2. read opinions of nodes  ####
        ####################################
        # try to read cache
        if read_cached:
            opinions_cached = False
            try:
                self.op_nodes = pd.read_pickle(self.PICKLE_PATH + pickle_opinions_filename)
                print("INFO: Read opinions from cache.")
                opinions_cached = True
            except FileNotFoundError:
                warnings.warn("No cached opinions found, read opinions from file.")
                opinions_cached = False

        # if not succeded with reading cache
        if not (read_cached and opinions_cached):
            print("read traits")
            self.op_nodes = pd.DataFrame()
            for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                nodes_t = pd.DataFrame(np.arange(0,self.NUMBER_NODES),columns=['node_id'])
                nodes_t = nodes_t.join(pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_opinions_t"+str(t)+".txt",header=None,names=['op_synthetic']))

                nodes_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                self.op_nodes = self.op_nodes.append(nodes_t)
            self.op_nodes["prev_op_synthetic"] = self.op_nodes["op_synthetic"]
            self.op_nodes["pop_op_synthetic"] = 1

            self.op_nodes.reset_index(inplace=True)
            self.op_nodes.to_pickle(self.PICKLE_PATH + pickle_opinions_filename)

        '''
        #### 3. Construct like-mindedness network out of opinions ####
        ##############################################################
        # try to read cache
        if read_cached:
            like_mindedness_cached = False
            try:
                self.d_ij = pd.read_pickle(self.PICKLE_PATH + pickle_like_mindedness_filename)
                print("INFO: Read like-mindedness from cache.")
                like_mindedness_cached = True
            except FileNotFoundError:
                warnings.warn("No cached like-mindedness net found, read like-mindedness from file.")
                like_mindedness_cached = False

        # if not succeded with reading cache
        if not (read_cached and like_mindedness_cached):
            print("calc likemindedness")
            self.d_ij = self._load_edges_from_opinions(self.op_nodes,self.time)
            self.d_ij.to_pickle(self.PICKLE_PATH + pickle_like_mindedness_filename)
        '''

    def load_null_model(self, a=0.8, b=0.2, read_cached = False):
        ''' load non-contagion null model '''
        # tc = time correction
        self.name = "null_model_a=" + str(a) + "_b=" + str(b) 
        if self.run != None:
            self.name += "_run=" + str(self.run)
        self.type = "SYN_NULL"

        pickle_relation_net_filename = self.RELATION_NET_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_opinions_filename = self.OPINIONS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_like_mindedness_filename = self.LIKE_MINDEDNESS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"

        self.file_name = f"null_model_{a}_{b}"

        if self.run != None: self.file_name += "_run" + str(self.run)

        print("Load World: " + self.name + " from " + self.file_name)

        # get number of time steps
        if not os.path.isdir(self.path +"/"+ self.file_name +"/"):
            raise ValueError('No model data found for given parameters at: ' + self.path +"/"+ self.file_name +"/")

        path, dirs, files = next(os.walk(self.path +"/"+ self.file_name +"/"))
        self.TIME_POINTS = floor(len(files)/2)

        print("Numer of time steps is: " + str(self.TIME_POINTS))

        ## 0. define time
        self.time = pd.date_range(start='1/1/2018', periods=self.TIME_POINTS/self.TIME_STEP, freq='d')
        self.time = pd.DataFrame(self.time.astype('str'),columns=['time'])

        ## 1. read relation network edges
        if read_cached:
            relations_cached = False
            try:
                self.a_ij = pd.read_pickle(self.PICKLE_PATH + pickle_relation_net_filename)
                print("INFO: Read relations from cache.")
                relations_cached = True
            except FileNotFoundError:
                warnings.warn("No cached relation network found, read relations from file.")
                relations_cached = False

        if not (read_cached and relations_cached):
            self.a_ij = pd.DataFrame()
            for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                edges_t = pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_edges_t"+str(t)+".txt",header=None,sep=' ',names=['id_A','id_B'])
                edges_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                edges_t['edge'] = True
                self.a_ij = self.a_ij.append(edges_t)
            self.a_ij.to_pickle(self.PICKLE_PATH + pickle_relation_net_filename)

        ## 2. read opinions of nodes
        if read_cached:
            opinions_cached = False
            try:
                self.op_nodes = pd.read_pickle(self.PICKLE_PATH + pickle_opinions_filename)
                print("INFO: Read opinions from cache.")
                opinions_cached = True
            except FileNotFoundError:
                warnings.warn("No cached opinions found, read opinions from file.")
                opinions_cached = False

        if not (read_cached and opinions_cached):
            self.op_nodes = pd.DataFrame()
            for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                nodes_t = pd.DataFrame(np.arange(0,self.NUMBER_NODES),columns=['node_id'])
                nodes_t = nodes_t.join(pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_opinions_t"+str(t)+".txt",header=None,names=['op_synthetic']))

                nodes_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                self.op_nodes = self.op_nodes.append(nodes_t)
            self.op_nodes["prev_op_synthetic"] = self.op_nodes["op_synthetic"]
            self.op_nodes["pop_op_synthetic"] = 1

            self.op_nodes.to_pickle(self.PICKLE_PATH + pickle_opinions_filename)
        '''
        ## 3. calculate like-mindedness network from op_nodes
        ## 3. Construct like-mindedness network out of opinions
        if read_cached:
            like_mindedness_cached = False
            try:
                self.d_ij = pd.read_pickle(self.PICKLE_PATH + pickle_like_mindedness_filename)
                print("INFO: Read like-mindedness from cache.")
                like_mindedness_cached = True
            except FileNotFoundError:
                warnings.warn("No cached like-mindedness net found, read like-mindedness from file.")
                like_mindedness_cached = False

        if not (read_cached and like_mindedness_cached):
            self.d_ij = self._load_edges_from_opinions(self.op_nodes,self.time)
            self.d_ij.to_pickle(self.PICKLE_PATH + pickle_like_mindedness_filename)
        '''

    def load_new_model(self, phi = 0, c = 0, d=0, read_cached = False):
        ''' load and process new model data '''
        # tc = time correction
        self.name = "new_model_phi=" + str(phi) + "_c=" + str(c) + "_d=" + str(d) 
        if self.run != None:
            self.name += "_run=" + str(self.run)
        self.type = "SYN_NEW"

        pickle_relation_net_filename = self.RELATION_NET_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_opinions_filename = self.OPINIONS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_like_mindedness_filename = self.LIKE_MINDEDNESS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"

        self.file_name = f"new_model_phi{phi}_c{c}_d{d}"

        if self.run != None: self.file_name += "_run" + str(self.run)
            
        self.file_name += "_tc"

        print("Load World: " + self.name + " from " + self.file_name)

        # get number of time steps
        if not os.path.isdir(self.path +"/"+ self.file_name +"/"):
            raise ValueError('No model data found for given parameters at: ' + self.path +"/"+ self.file_name +"/")

        path, dirs, files = next(os.walk(self.path +"/"+ self.file_name +"/"))
        self.TIME_POINTS = floor(len(files)/2)

        print("Numer of time steps is: " + str(self.TIME_POINTS))

        ## 0. define time
        self.time = pd.date_range(start='1/1/2018', periods=self.TIME_POINTS/self.TIME_STEP, freq='d')
        self.time = pd.DataFrame(self.time.astype('str'),columns=['time'])

        ## 1. read relation network edges
        if read_cached:
            relations_cached = False
            try:
                self.a_ij = pd.read_pickle(self.PICKLE_PATH + pickle_relation_net_filename)
                print("INFO: Read relations from cache.")
                relations_cached = True
            except FileNotFoundError:
                warnings.warn("No cached relation network found, read relations from file.")
                relations_cached = False

        if not (read_cached and relations_cached):
            self.a_ij = pd.DataFrame()
            for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                edges_t = pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_edges_t"+str(t)+".txt",header=None,sep=' ',names=['id_A','id_B'])
                edges_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                edges_t['edge'] = True
                self.a_ij = self.a_ij.append(edges_t)
            self.a_ij.to_pickle(self.PICKLE_PATH + pickle_relation_net_filename)

        ## 2. read opinions of nodes
        if read_cached:
            opinions_cached = False
            try:
                self.op_nodes = pd.read_pickle(self.PICKLE_PATH + pickle_opinions_filename)
                print("INFO: Read opinions from cache.")
                opinions_cached = True
            except FileNotFoundError:
                warnings.warn("No cached opinions found, read opinions from file.")
                opinions_cached = False

        if not (read_cached and opinions_cached):
            self.op_nodes = pd.DataFrame()
            for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                nodes_t = pd.DataFrame(np.arange(0,self.NUMBER_NODES),columns=['node_id'])
                nodes_t = nodes_t.join(pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_opinions_t"+str(t)+".txt",header=None,names=['op_synthetic']))

                nodes_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                self.op_nodes = self.op_nodes.append(nodes_t)
            self.op_nodes["prev_op_synthetic"] = self.op_nodes["op_synthetic"]
            self.op_nodes["pop_op_synthetic"] = 1

            self.op_nodes.to_pickle(self.PICKLE_PATH + pickle_opinions_filename)

        '''  Until further notic outdated
        ## 3. calculate like-mindedness network from op_nodes
        ## 3. Construct like-mindedness network out of opinions
        if read_cached:
            like_mindedness_cached = False
            try:
                self.d_ij = pd.read_pickle(self.PICKLE_PATH + pickle_like_mindedness_filename)
                print("INFO: Read like-mindedness from cache.")
                like_mindedness_cached = True
            except FileNotFoundError:
                warnings.warn("No cached like-mindedness net found, read like-mindedness from file.")
                like_mindedness_cached = False

        if not (read_cached and like_mindedness_cached):
            self.d_ij = self._load_edges_from_opinions(self.op_nodes,self.time)
            self.d_ij.to_pickle(self.PICKLE_PATH + pickle_like_mindedness_filename)
        '''
        
    def load_sis_model(self, nu = 0, delta=0, read_cached = False):
        ''' load and process sis model data '''
        # tc = time correction
        self.name = "sis_model" + "_nu=" + str(nu) + "_delta=" + str(delta) 
        if self.run != None:
            self.name += "_run=" + str(self.run)
        self.type = "SYN_SIS"

        pickle_relation_net_filename = self.RELATION_NET_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_opinions_filename = self.OPINIONS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"
        pickle_like_mindedness_filename = self.LIKE_MINDEDNESS_PICKLE \
                                       + "_" + self.name \
                                       + ".pkl"

        self.file_name = f"sis_model_nu{nu}_delta{delta}"

        if self.run != None: self.file_name += "_run" + str(self.run)
            
        print("Load World: " + self.name + " from " + self.file_name)

        # get number of time steps
        if not os.path.isdir(self.path +"/"+ self.file_name +"/"):
            raise ValueError('No model data found for given parameters at: ' + self.path +"/"+ self.file_name +"/")

        path, dirs, files = next(os.walk(self.path +"/"+ self.file_name +"/"))
        self.TIME_POINTS = floor(len(files)/2)

        print("Numer of time steps is: " + str(self.TIME_POINTS))

        ## 0. define time
        self.time = pd.date_range(start='1/1/2018', periods=self.TIME_POINTS/self.TIME_STEP, freq='d')
        self.time = pd.DataFrame(self.time.astype('str'),columns=['time'])

        ## 1. read relation network edges
        if read_cached:
            relations_cached = False
            try:
                self.a_ij = pd.read_pickle(self.PICKLE_PATH + pickle_relation_net_filename)
                print("INFO: Read relations from cache.")
                relations_cached = True
            except FileNotFoundError:
                warnings.warn("No cached relation network found, read relations from file.")
                relations_cached = False

        if not (read_cached and relations_cached):
            self.a_ij = pd.DataFrame()
            for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                edges_t = pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_edges_t"+str(t)+".txt",header=None,sep=' ',names=['id_A','id_B'])
                edges_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                edges_t['edge'] = True
                self.a_ij = self.a_ij.append(edges_t)
            self.a_ij.to_pickle(self.PICKLE_PATH + pickle_relation_net_filename)

        ## 2. read opinions of nodes
        if read_cached:
            opinions_cached = False
            try:
                self.op_nodes = pd.read_pickle(self.PICKLE_PATH + pickle_opinions_filename)
                print("INFO: Read opinions from cache.")
                opinions_cached = True
            except FileNotFoundError:
                warnings.warn("No cached opinions found, read opinions from file.")
                opinions_cached = False

        if not (read_cached and opinions_cached):
            self.op_nodes = pd.DataFrame()
            for t in range(0,self.TIME_POINTS,self.TIME_STEP):
                nodes_t = pd.DataFrame(np.arange(0,self.NUMBER_NODES),columns=['node_id'])
                nodes_t = nodes_t.join(pd.read_csv(self.path +"/"+ self.file_name +"/"+ self.file_name +"_opinions_t"+str(t)+".txt",header=None,names=['op_synthetic']))

                nodes_t['time'] = self.time.time[int(t/self.TIME_STEP)]
                self.op_nodes = self.op_nodes.append(nodes_t)
            self.op_nodes["prev_op_synthetic"] = self.op_nodes["op_synthetic"]
            self.op_nodes["pop_op_synthetic"] = 1

            self.op_nodes.to_pickle(self.PICKLE_PATH + pickle_opinions_filename)
            
        
    def load_marc_model(self, contagion = False):
        ''' load and process sis model data '''
        # tc = time correction
        self.name = "marc_model"
        
        self.type = "SYN_M"
        
        if contagion == False:
            self.file_name = "marcs_model/new_model_only_exploration.p"
        else:
            self.file_name = "marcs_model/new_model_contagion_and_exploration.p"
            
        print("Load World: " + self.name + " from " + self.file_name)

        self.TIME_POINTS = 1000

        print("Numer of time steps is: " + str(self.TIME_POINTS))

        ## 0. define time
        self.time = pd.date_range(start='1/1/2018', periods=self.TIME_POINTS, freq='d')
        self.time = pd.DataFrame(self.time.astype('str'),columns=['time'])
            
            
        data = pd.read_pickle(self.file_name)
        self.op_nodes = pd.DataFrame(data["activity"]).unstack().to_frame("op_synthetic").reset_index().rename(columns={"level_0":"node_id", "level_1":"time"})
        self.op_nodes.time = list(map(lambda i: self.time.time.iloc[i], self.op_nodes.time))

        links = pd.DataFrame(data["links"],columns=["id_A","id_B"])
        a_ij_list = []
        for t in range(1000):
            a_ij_tmp = links.copy()
            a_ij_tmp["time"] = self.time.time.iloc[t]
            a_ij_list += [a_ij_tmp]
        self.a_ij = pd.concat(a_ij_list)
        self.a_ij["edge"] = True
