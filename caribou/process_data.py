#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo update doc string

###########
# Imports #
###########

import json
import tqdm
import datetime
import random
import matplotlib.cm
import pandas as pd
import numpy as np
import multiprocessing as mp
from pandarallel import pandarallel
from typing import Tuple

from misc_utilities import *

import bokeh.layouts
import bokeh.plotting
import bokeh.models
import bokeh.tile_providers

# @todo update these imports

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

OUTPUT_HTML_FILE = 'output.html'

# https://www.kaggle.com/jessemostipak/caribou-location-tracking
LOCATIONS_CSV_FILE = './data/locations.csv'
INDIVIDUALS_CSV_FILE = './data/individuals.csv'

##################################
# Application Specific Utilities #
##################################

WGS84_K = 6378137

def wgs84_long_to_web_mercator_x(longitude: float) -> float:
    x = longitude * WGS84_K * np.pi/180.0
    return x

def wgs84_lat_to_web_mercator_y(latitude: float) -> float:
    y = np.log(np.tan((90 + latitude) * np.pi/360.0)) * WGS84_K
    return y

###################
# Data Processing #
###################

def process_data(locations_df: pd.DataFrame) -> pd.DataFrame:
    locations_df['longitude_x'] = locations_df.longitude.parallel_map(wgs84_long_to_web_mercator_x)
    locations_df['latitude_y'] = locations_df.latitude.parallel_map(wgs84_lat_to_web_mercator_y)
    locations_df['date'] = locations_df.timestamp.parallel_map(datetime.datetime.date)
    return locations_df

#################
# Visualization #
#################

def _initialize_map_figure(locations_df: pd.DataFrame) -> bokeh.plotting.Figure:    
    tile_provider = bokeh.tile_providers.get_provider(bokeh.tile_providers.ESRI_IMAGERY)
    min_longitude_x = locations_df.longitude_x.min()
    max_longitude_x = locations_df.longitude_x.max()
    min_latitude_y = locations_df.latitude_y.min()
    max_latitude_y = locations_df.latitude_y.max()
    map_figure = bokeh.plotting.figure(
        plot_width=1600, 
        plot_height=800,
        x_range=(min_longitude_x, max_longitude_x),
        y_range=(min_latitude_y, max_latitude_y),
        x_axis_type='mercator',
        y_axis_type='mercator',
    )
    map_figure.sizing_mode = 'scale_width'
    map_figure.add_layout(bokeh.models.Title(text='Movement of 260 Caribou from 1988 to 2016', align='center'), 'above')
    map_figure.add_tile(tile_provider)
    return map_figure

def _generate_multi_line_data_source_df(animal_id_groupby: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    xs_series = animal_id_groupby.apply(lambda group: group.longitude_x.tolist()).rename('xs') # parallel_apply slower
    ys_series = animal_id_groupby.apply(lambda group: group.latitude_y.tolist()).rename('ys') # parallel_apply slower
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, animal_id_groupby.ngroups))
    colors = eager_map(lambda rgb_triple: bokeh.colors.RGB(*rgb_triple), colors * 255)
    random.seed(0)
    random.shuffle(colors)
    multi_line_data_source_df = xs_series.to_frame().join(ys_series.to_frame())
    multi_line_data_source_df['color'] = pd.Series(colors, index=xs_series.index)
    return multi_line_data_source_df

def _draw_caribou_lines(multi_line_data_source_df: pd.DataFrame, map_figure: bokeh.plotting.Figure) -> None:
    multi_line_data_source = bokeh.models.ColumnDataSource(multi_line_data_source_df)
    map_figure.multi_line(xs='xs', ys='ys', line_color='color', source=multi_line_data_source, line_width=2, line_alpha=0.25)
    return

def _generate_location_by_date(locations_df: pd.DataFrame) -> dict:
    locations_df = locations_df[['animal_id', 'timestamp', 'date', 'longitude_x', 'latitude_y']]
    indices_of_earliest_timestamp_for_animal_and_date = locations_df.groupby(['animal_id', 'date'])['timestamp'].idxmin()
    locations_df = locations_df.iloc[indices_of_earliest_timestamp_for_animal_and_date]
    locations_df = locations_df[['animal_id', 'date', 'longitude_x', 'latitude_y']]
    
    manager = mp.Manager()
    location_by_date = manager.dict()    
    def _update_location_by_date(group: pd.DataFrame) -> None:
        assert len(group.date.unique() == 1)
        date = group.date.iloc[0]
        date_string = date.isoformat()
        location_by_date[date_string] = group[['animal_id', 'longitude_x', 'latitude_y']].set_index('animal_id').to_dict(orient='index')
        return
    locations_df.groupby(['date']).parallel_apply(_update_location_by_date)
        
    location_by_date = dict(location_by_date)
    return location_by_date

def _generate_earliest_and_latest_date_by_animal_id_dict(locations_df: pd.DataFrame) -> dict:
    earliest_latest_date_df = locations_df.groupby(['animal_id']).agg({'date': ['min', 'max']})
    earliest_latest_date_df.columns = earliest_latest_date_df.columns.get_level_values(1)
    earliest_latest_date_df.rename(columns={'min': 'latest_date', 'max': 'earliest_date'}, inplace=True)
    earliest_latest_date_df = earliest_latest_date_df.parallel_applymap(datetime.date.isoformat)
    earliest_and_latest_date_by_animal_id_dict = earliest_latest_date_df.to_dict(orient='index')
    return earliest_and_latest_date_by_animal_id_dict

def _generate_date_slider(locations_df: pd.DataFrame, multi_line_data_source_df: pd.DataFrame, animal_id_groupby: pd.core.groupby.generic.DataFrameGroupBy, map_figure: bokeh.plotting.Figure) -> bokeh.models.DateSlider:
    start_date = locations_df.timestamp.min().date()
    end_date = locations_df.timestamp.max().date()
    
    date_slider = bokeh.models.DateSlider(start=start_date, end=end_date, value=start_date, step=1, title='Date', align='center')
    location_by_date = _generate_location_by_date(locations_df)
    earliest_and_latest_date_by_animal_id_dict = _generate_earliest_and_latest_date_by_animal_id_dict(locations_df)
    with open('./slider_callback.js', 'r') as f:
        js_callback_code = f.read()
    animal_ids = locations_df.animal_id.unique().tolist()
    date_slider_callback = bokeh.models.callbacks.CustomJS(
        args=dict(
            dateSlider=date_slider, locationByDate=location_by_date, earliestAndLatestDateByAnimalId=earliest_and_latest_date_by_animal_id_dict, animalIds=animal_ids
        ),
        code=js_callback_code)
    date_slider.js_on_change('value', date_slider_callback)
    return date_slider

def create_output_html(locations_df: pd.DataFrame) -> None:
    bokeh.plotting.output_file(OUTPUT_HTML_FILE, mode='inline')
    map_figure = _initialize_map_figure(locations_df)
    animal_id_groupby = locations_df.groupby('animal_id').parallel_apply(lambda group: group.set_index('timestamp').sort_index()[['longitude_x', 'latitude_y']]).groupby('animal_id')
    multi_line_data_source_df = _generate_multi_line_data_source_df(animal_id_groupby)
    _draw_caribou_lines(multi_line_data_source_df, map_figure)
    date_slider = _generate_date_slider(locations_df, multi_line_data_source_df, animal_id_groupby, map_figure)
    layout = bokeh.layouts.column(map_figure, date_slider)
    bokeh.plotting.save(layout, title='Caribou Movement')
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    locations_df = pd.read_csv(LOCATIONS_CSV_FILE, parse_dates=['timestamp'])
    locations_df = process_data(locations_df)
    create_output_html(locations_df)
    return

if __name__ == '__main__':
    main()
 
