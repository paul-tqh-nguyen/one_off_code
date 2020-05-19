#!/usr/bin/python3
'#!/usr/bin/python3 -OO' # @ todo use this

'''
'''

# @todo fill in the doc string

###########
# Imports #
###########

import pandas as pd
from typing import Tuple

from misc_utilities import *

from pyproj import Transformer
from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import figure
from bokeh.io import export_png

#################
# Functionality #
#################

def lat_long_to_mercator(lat_arg: float, long_arg: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs('epsg:4326', 'epsg:3857')
    mercator_x, mercator_y = transformer.transform(long_arg, lat_arg)
    return mercator_x, mercator_y

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    airports_df = pd.read_csv('./airports.dat', header=None, names=['Airport_ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz_database_time_zone', 'Type', 'Source'])
    
    m = figure(plot_width=1600, 
               plot_height=800,
               x_range=(-10_000_000, 10_000_000),
               y_range=(-12_000_000, 12_000_000),
               x_axis_type='mercator', 
               y_axis_type='mercator')

    tile_provider = get_provider(Vendors.ESRI_IMAGERY)    
    m.add_tile(tile_provider)
    
    m.circle(x=detroit_mercator_x, y=detroit_mercator_y, size=10, color='red')
    m.circle(x=cleveland_mercator_x, y=cleveland_mercator_y, size=10, color='red')
    m.circle(x=chicago_mercator_x, y=chicago_mercator_y, size=10, color='red')

    export_png(m, filename="map.png")
    
    return

if __name__ == '__main__':
    main()
