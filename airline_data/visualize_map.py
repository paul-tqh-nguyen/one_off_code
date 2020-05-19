#!/usr/bin/python3
'#!/usr/bin/python3 -OO' # @ todo use this

'''
'''

# @todo fill in the doc string

###########
# Imports #
###########

from pyproj import Transformer
from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import figure
from bokeh.io import export_png
from typing import Tuple

from misc_utilities import *

#################
# Functionality #
#################

def lat_long_to_mercator(long_arg: float, lat_arg: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs('epsg:4326', 'epsg:3857')
    mercator_x, mercator_y = transformer.transform(lat_arg, long_arg)
    return mercator_x, mercator_y

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    # Detroit
    detroit_lat = -83.047752
    detroit_long = 42.334197
    detroit_mercator_x, detroit_mercator_y = lat_long_to_mercator(detroit_lat, detroit_long)
    # Cleveland
    cleveland_lat = -81.694703
    cleveland_long = 41.499437
    cleveland_mercator_x, cleveland_mercator_y = lat_long_to_mercator(cleveland_lat, cleveland_long)
    # Chicago
    chicago_lat = -87.629849
    chicago_long = 41.878111
    chicago_mercator_x, chicago_mercator_y = lat_long_to_mercator(chicago_lat, chicago_long)
    
    m = figure(plot_width=1600, 
               plot_height=800,
               x_range=(-12_000_000, 9_000_000),
               y_range=(-1_000_000, 7_000_000),
               x_axis_type='mercator', 
               y_axis_type='mercator')

    tile_provider = get_provider(Vendors.CARTODBPOSITRON)    
    m.add_tile(tile_provider)
    
    m.circle(x=detroit_mercator_x, y=detroit_mercator_y, size=10, color='red')
    m.circle(x=cleveland_mercator_x, y=cleveland_mercator_y, size=10, color='red')
    m.circle(x=chicago_mercator_x, y=chicago_mercator_y, size=10, color='red')

    export_png(m, filename="map.png")
    
    return

if __name__ == '__main__':
    main()
