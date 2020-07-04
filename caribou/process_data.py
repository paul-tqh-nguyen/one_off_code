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
import bokeh.plotting
import bokeh.tile_providers
import pandas as pd
import numpy as np
import multiprocessing as mp
from pandarallel import pandarallel
from typing import Tuple

from misc_utilities import *

# @todo update these imports

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

OUTPUT_HTML_FILE = 'output.html'

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
    return locations_df

##########
# Driver #
##########

def create_output_html(locations_df: pd.DataFrame) -> None:
    bokeh.plotting.output_file(OUTPUT_HTML_FILE)
    tile_provider = bokeh.tile_providers.get_provider(bokeh.tile_providers.ESRI_IMAGERY)
    
    min_longitude_x = locations_df.longitude_x.min()
    max_longitude_x = locations_df.longitude_x.max()
    min_latitude_y = locations_df.latitude_y.min()
    max_latitude_y = locations_df.latitude_y.max()

    figure = bokeh.plotting.figure(
        plot_width=1600, 
        plot_height=800,
        x_range=(min_longitude_x, max_longitude_x),
        y_range=(min_latitude_y, max_latitude_y),
        x_axis_type='mercator',
        y_axis_type='mercator',
        # sizing_mode='scale_both',
    )
    figure.add_tile(tile_provider)
    
    animal_id_groupby = locations_df.groupby('animal_id')
    xs = animal_id_groupby.apply(lambda group: group.set_index('timestamp').sort_index().longitude_x.tolist()).tolist()
    ys = animal_id_groupby.apply(lambda group: group.set_index('timestamp').sort_index().latitude_y.tolist()).tolist()
    number_of_unique_animal_ids = len(locations_df.animal_id.unique())
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, number_of_unique_animal_ids))
    colors = eager_map(lambda rgb_triple: bokeh.colors.RGB(*rgb_triple), colors * 255)
    random.seed(0)
    random.shuffle(colors)
    figure.multi_line(xs, ys, line_width=2, line_alpha=0.25, line_color=colors)

    bokeh.plotting.save(figure)
    return

@debug_on_error
def main() -> None:
    locations_df = pd.read_csv(LOCATIONS_CSV_FILE, parse_dates=['timestamp'])
    locations_df = process_data(locations_df)
    create_output_html(locations_df)
    return

if __name__ == '__main__':
    main()
 
