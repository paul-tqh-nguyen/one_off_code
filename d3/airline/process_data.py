#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo update doc string

###########
# Imports #
###########

import json
from typing import List

from misc_utilities import *

###########
# Globals #
###########

# https://github.com/holtzy/D3-graph-gallery/blob/master/DATA/world.geojson
WORLD_GEOJSON_FILE_LOCATION = './data/world.geojson'

OUTPUT_GEOJSON_FILE_LOCATION = './data/processed_data.geojson'

#####################
# General Utilities #
#####################

def create_geojson_data_from_features(features: list) -> dict:
    return {"type": "FeatureCollection", "features": features}

def generate_line_path_feature(information_type: str, coordinates: list) -> dict:
    return {
        "type":"Feature",
        "properties": {"information-type": information_type},
        "geometry": {
	    "type": "LineString",
	    "coordinates": coordinates}
    }

############################
# Data Gathering Utilities #
############################

def generate_landmass_features() -> List[dict]:
    with open(WORLD_GEOJSON_FILE_LOCATION, 'r') as file_handle:
        world_geojson_data = json.load(file_handle)
    usa_feature = only_one([feature for feature in world_geojson_data['features'] if feature['properties']['name']=='USA'])
    usa_feature['properties'] = {"information-type": "landmass"}
    del usa_feature['id']
    landmass_features = [usa_feature]
    return landmass_features

def generate_all_flight_path_features() -> List[dict]:
    # @ todo finish this
    return 

##########
# Driver #
##########

@debug_on_error
def process_data() -> None:
    features = generate_landmass_features() + generate_all_flight_path_features()
    final_geojson_data = create_geojson_data_from_features(features)
    with open(OUTPUT_GEOJSON_FILE_LOCATION, 'w') as file_handle:
        json.dump(final_geojson_data, file_handle)
    return

if __name__ == '__main__':
    process_data()
 
