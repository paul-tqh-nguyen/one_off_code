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
import pandas as pd
import numpy as np
import multiprocessing as mp
from pandarallel import pandarallel

from misc_utilities import *

# @todo update these imports

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

###################
# Data Processing #
###################

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    from bokeh.plotting import figure, output_file, show
    from bokeh.tile_providers import CARTODBPOSITRON, get_provider
    
    output_file("tile.html")
    
    tile_provider = get_provider(CARTODBPOSITRON)
    
    # range bounds supplied in web mercator coordinates
    p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
               x_axis_type="mercator", y_axis_type="mercator")
    p.add_tile(tile_provider)
    
    show(p)
    return

if __name__ == '__main__':
    main()
 
