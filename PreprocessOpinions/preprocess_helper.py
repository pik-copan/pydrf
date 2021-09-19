import pandas as pd
import numpy as np


class Helper:
    
    @staticmethod          
    def _smooth_with_gaussian(people_slice, opinion_type, location, sigma_t):
        people_slice = people_slice.copy()
        people_slice.timestamp = people_slice.timestamp.astype('int')/1000000000.0 # to seconds
        time_matrix = np.array([people_slice.timestamp.values]*len(people_slice.timestamp))
        diff = (time_matrix - time_matrix.transpose()) #seconds
        sigma = pd.to_timedelta(sigma_t, unit='d').total_seconds() #seconds
        two_sigma_sqr = 2* sigma * sigma
        matrix = np.exp(-(diff * diff)/two_sigma_sqr)
        #sigma_t = pd.to_timedelta(sigma_t, unit='d').total_seconds() #seconds
        #t_0 = abs(sigma_t / np.log(0.5)) # choose t_0 so that the half-life period is sigma_t
        #matrix = np.exp(diff/t_0) 
        filter_past = np.tril(np.ones_like(matrix))
        matrix *= filter_past
        people_slice[opinion_type + '_abs'] =  np.dot(matrix, people_slice['next_to_' + location].astype("int"))
        people_slice[opinion_type] =  np.dot(matrix, people_slice['next_to_' + location].astype("int")) / np.sum(matrix, axis = 1)
        return people_slice.set_index("timestamp")
           
    @staticmethod
    def count_visits_per_time(data, location, opinion_type, time = 7, min_visit_time_minutes = 15):
        
        # set "next_to"-variable to True if there is a location in range
        data.loc[[isinstance(x, bool) == False for x in data['next_to_' + location]], 'next_to_' + location] = True
        
        # calc time fore merge
        start = pd.to_datetime(data.timestamp.min(), unit='ms')
        stop = pd.to_datetime(data.timestamp.max(), unit='ms')
        timeline = pd.DataFrame(pd.date_range(start, stop, freq='D'),columns=['timestamp'])
        
        # merge missing time steps and fill missing values with False
        data.timestamp = pd.to_datetime(data.timestamp, unit='ms')
        data = pd.merge_ordered(data, timeline, left_by='user', how='outer')
        data['next_to_' + location] = data['next_to_' + location].fillna(False)
                
        # people have to stay min min_visit_time_minutes
        data.loc[(data.departure - data.arrival) < min_visit_time_minutes * 60, 'next_to_' + location] = False
        
        # aggregate to days
        data.timestamp = pd.to_datetime(data.timestamp, unit='ms').dt.to_period('D').dt.to_timestamp()
        # aggregate all entries of a day to one entry
        data = data.groupby(['timestamp','user']).any().reset_index()
        
        raw_data = data.copy()
        
        # count visits per time and smoothen with gaussian
        data[opinion_type] = 0 # init column
        data = data.groupby('user')['timestamp', 'next_to_' + location, opinion_type].apply( \
                                                lambda p: Helper._smooth_with_gaussian(p, opinion_type, location, time) \
                                                                                                 ).reset_index()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s').dt.to_period('D').dt.to_timestamp()
        
        # create nice df
        data = data[['user', 'timestamp', opinion_type, opinion_type + '_abs']].copy()
        data.rename(columns={'timestamp':'time', 'user':'node_id'}, inplace=True)
        #data.time = data.time.dt.to_period('W-MON').dt.to_timestamp()
        #data.time = data.time - pd.to_timedelta(1, unit='d')
        #data = data.groupby(["node_id",'time']).mean()

        return data, raw_data
