import pandas as pd
import numpy as np


class Helper:
               
    @staticmethod
    def filter_raw_data(data, time_begin, time_end):
        # apply filters
        # filter out empty scans
        data = data[data.bt_mac.astype('str') != '-1'] 
        # filter out non study participants
        data = data[data.bt_mac > 0]
        # filter connection with rssi < 80 (signal strength)
        data = data[data.rssi.astype('int') < -80]
        # filter only important columns
        data = data[['bt_mac','rssi','timestamp','user']]
        # apply name conventions
        data.rename(columns={'user':'id_A','bt_mac':'id_B','timestamp':'time'},inplace=True)
        data = data[(data.id_A < 1000) & (data.id_B < 1000)] #clean strange users

        #take only data in certain time range
        data.time = pd.to_datetime(data.time,unit='ms')
        data = data[(data.time > time_begin) & (data.time < time_end)]
        
        # take both directions id_A->id_B, id_B->id_A
        data_inv = data.rename(columns={'id_A':'id_B','id_B':'id_A'})
        data = pd.concat([data, data_inv])
        data.drop_duplicates(['id_A','id_B','time'],inplace=True)
        # dont count edges twice
        data = data[data.id_A < data.id_B]
        
        return data
    
    @staticmethod
    def agg_to_visits_per_day(proxi, time, verbose=True):
    
        proxi.time = proxi.time.dt.to_period('d').dt.to_timestamp()
        #proxi.time = proxi.time - pd.to_timedelta(1, unit='d')
        
        # count encounters per day
        proxi['encounter'] = 1
        proxi = proxi.groupby(['id_A','id_B','time']).encounter.sum().reset_index()
        
        #insert time steps with no recorded encounter
        proxi = proxi.groupby(['id_A','id_B'])[['time','encounter']] \
                .apply( lambda p: \
                       pd.DataFrame(p).set_index(['time']).join(time.set_index(['time']), how='outer') \
                      )
        proxi.reset_index(inplace=True)

        # fill unknown encounters with 0
        proxi.fillna(0,inplace=True)
        
        return proxi
    
    @staticmethod
    def agg_to_visits_per_week(proxi):
    
        sigma = pd.to_timedelta(6, unit='d').total_seconds()
        two_sigma_sqr = 2* sigma * sigma

        # weighted sum over a week
        proxi["encounter_per_week"] = 0
        proxi = proxi.groupby(['id_A','id_B'])['time','encounter', "encounter_per_week"].apply(lambda p: Helper._smooth_with_gaussian(p, two_sigma_sqr))
        
        proxi.reset_index(inplace=True)
        proxi.time = pd.to_datetime(proxi.time, unit='s')
        '''
        proxi.time = pd.to_datetime(proxi.time, unit='s').dt.to_period('W-MON').dt.to_timestamp()
        proxi.time = proxi.time - pd.to_timedelta(1, unit='d')
        proxi = proxi.groupby(['id_A','id_B','time']).mean()
        '''
        return proxi

    @staticmethod
    def _smooth_with_gaussian(proxi_slice, two_sigma_sqr):
        proxi = proxi_slice.copy()
        proxi.time = proxi.time.astype('int')/1000000000.0 # to seconds
        time_matrix = np.array([proxi.time.values]*len(proxi.time))
        diff = time_matrix - time_matrix.transpose()
        matrix = np.exp(-(diff * diff)/two_sigma_sqr)
        filter_past = np.tril(np.ones_like(diff))
        matrix *= filter_past
        proxi.encounter_per_week =  np.dot(matrix, proxi.encounter)
        return proxi.set_index('time')