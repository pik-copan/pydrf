from geolocation_tools.geo_tools.locations import Locations, Helper
from sensible_raw.loaders import loader
import pandas as pd
import sys
import datetime


dfs = []
for i in ['september_2013','october_2013','november_2013','december_2013',
          'january_2014','february_2014','march_2014','april_2014',
          'may_2014','june_2014','july_2014','august_2014',
         'september_2014','october_2014','november_2014','december_2014',
          'january_2015','february_2015','march_2015','april_2015',
          'may_2015','june_2015','july_2015','august_2015',
         'september_2015']:
    df = loader.load_data("stop_locations", i, as_dataframe=True) # data range september_2013 to september_2015
    dfs.append(df)
df = pd.concat(dfs)

loc_type = sys.argv[1]

# load data
if loc_type == 'burgers':
    loc = Locations(path='geolocation_tools/data/burgers.geojson', name='burger places')
elif loc_type == 'sports':
    loc = Locations(path='geolocation_tools/data/sports_and_fitness.geojson', name='sports places')
elif loc_type == 'grocery':
    loc = Locations(path='geolocation_tools/data/grocery_stores.geojson', name='grocery stores')
elif loc_type == 'vegies':
    loc = Locations(path='geolocation_tools/data/fruit_vegies.geojson', name='green groceries')

df['next_to_' + loc_type] = pd.Series([loc.any_next_to( [x[1]['lon'], x[1]['lat']], 0.01) for x in df[['lat','lon']].iterrows()])

now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H:%M")
df.to_pickle("tmp/" + loc_type + "_" + now + ".pkl")