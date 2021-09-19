from typing import List

import geopandas as gpd
import shapely.geometry as gm
from bokeh.models import ColumnDataSource
import pandas as pd
import numpy as np
from math import radians, sin, cos, acos


class Helper:

    @staticmethod
    def separate_polygons_and_points(raw, geom='geometry'):
        """separate polygons and points in geopandas dataframes"""

        pts_map = [type(b) == gm.Point for b in raw[geom]]
        ply_map = [type(b) == gm.Polygon for b in raw[geom]]

        return raw.loc[pts_map], raw.loc[ply_map]

    @staticmethod
    def get_point_coordinates(row, geom, coord_type):
        """Calculates coordinates ('x' or 'y') of a Point geometry"""
        if coord_type == 'x':
            return row[geom].x
        elif coord_type == 'y':
            return row[geom].y

    @staticmethod
    def get_polygon_coordinates(row, geom, coord_type):
        """Returns the coordinates ('x' or 'y') of edges of a Polygon exterior"""

        # Parse the exterior of the coordinate
        exterior = row[geom].exterior

        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return list(exterior.coords.xy[0])
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return list(exterior.coords.xy[1])

    def add_point_layer(self, figure, data, color='red', size=7, name=''):
        """from a geopandas frame, add a layer of points to an existing figure"""

        if len(data) is 0:
            return

        for c in ['x', 'y']:
            data[c] = data.apply(self.get_point_coordinates, geom='geometry', coord_type=c, axis=1)

        coords = data.drop('geometry', axis=1).copy()
        psource = ColumnDataSource(coords)
        figure.circle('x', 'y', source=psource,
                      color=color, size=size, legend=name, muted_alpha=0.1, alpha=0.6)

    def add_polygon_layer(self, figure, data, color='red', name=''):
        """from a geopandas frame, add a layer of polygons to an existing figure"""

        if len(data) is 0:
            return

        for c in ['x', 'y']:
            data[c] = data.apply(self.get_polygon_coordinates, geom='geometry', coord_type=c, axis=1)

        coords = data.drop('geometry', axis=1).copy()
        psource = ColumnDataSource(coords)
        figure.patches('x', 'y', source=psource,
                       color=color,
                       legend=name, muted_alpha=0.1, alpha=0.6)

    @staticmethod
    def add_point(figure, point, color='red', size=7, name='', alpha=0.6):
        """from a geopandas frame, add a layer of points to an existing figure"""

        if type(point) == pd.DataFrame:
            point_source = ColumnDataSource(point)
        else:
            if type(point) == list and len(point) == 2:
                data = {'name': name, 'x': point[0], 'y': point[1]}
            else:
                try:
                    data = {'name': name, 'x': point.x, 'y': point.y}
                except AttributeError:
                    data = {'name': name, 'x': point.geometry.x, 'y': point.geometry.y}

            df = pd.DataFrame(data=data, index=np.arange(1))
            point_source = ColumnDataSource(df)

        figure.circle('x', 'y', source=point_source,
                      color=color, size=size, legend=name, muted_alpha=0.1, alpha=alpha)

    @staticmethod
    def add_trajectory(figure, trajectory, color='blue', name='', alpha=0.5):

        psource = ColumnDataSource(trajectory)
        figure.line('x', 'y', source=psource,
                    color=color, legend=name, muted_alpha=0.1, alpha=alpha)


    @staticmethod
    def find_locations(trajectory: pd.DataFrame, places: list, distance):
        """classify trajectory with places

        Parameters:
        ----------
        trajectory: pd.DataFrame
            trajectory with columns named 'x' and 'y' containing coordinates.
        places: list
            list of Locations objects with respect to which proximity is measured
        distance: float
            distance threshold for 'checkin' to place

        Returns:
        -------
        df: pd.Dataframe
            dataframe like the input trajectory with additional
            column containing the checked in places
        """

        assert all([col in trajectory.columns.values for col in ['x', 'y']]), 'trajectory must have x and y columns'
        assert len(places) > 0, 'places must contain at least one instance of Location objects'

        df = trajectory

        for i, row in trajectory.iterrows():
            names = ''
            for p in places:
                d, place = p.closest_to(list(row[['x', 'y']]))
                if d < distance:
                    names += p.name + ', '
            df.loc[i, 'place'] = names[:-2]
        return df


class Locations(Helper):
    """helps to work with sets of locations"""

    def __init__(self, path: str, name: str):
        self.name = name
        self.all_locations = gpd.read_file(path)
        self.pts, self.ply = self.separate_polygons_and_points(self.all_locations)

    def plot(self, figure, color='red', size=7, data=None):
        """plot all locations to bokeh figure"""
        pts, ply = self.separate_polygons_and_points(data)
        self.add_point_layer(figure, pts, color=color, size=size, name=self.name)
        self.add_polygon_layer(figure, ply, color=color, name=self.name)

    def closest_to(self, x: List[float]):
        """return location that is closest to x"""

        xy1 = gm.Point(x)

        distances = [xy1.distance(x) for x in self.all_locations['geometry']]
        min_distance = min(distances)
        closest_location = self.all_locations.iloc[distances.index(min_distance)]

        return min_distance, closest_location
    
    def any_next_to(self, x: List[float], max_distance: float):
        """return if any location is next to x"""

        xy1 = gm.Point(x)

        distances = [xy1.distance(x) for x in self.all_locations['geometry']]
        min_distance = min(distances)
        
        if min_distance < max_distance:
            return self.all_locations.iloc[distances.index(min_distance)]
        else:
            return False
        