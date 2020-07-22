'#!/usr/bin/python3 -OO' # @todo make this the default

'''
'''
# @todo update doc string

###########
# Imports #
###########

import json
import multiprocessing as mp
import pandas as pd
import multiprocessing as mp
from pandarallel import pandarallel
from shapely.geometry import Point, Polygon
from typing import Union

from misc_utilities import *

# @todo make sure these are used

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

CRASH_DATA_CSV_FILE_LOCATION = './data/Motor_Vehicle_Collisions_-_Crashes.csv'
BOROUGH_GEOJSON_FILE_LOCATION = './data/Borough Boundaries.geojson'
ZIP_CODE_GEOJSON_FILE_LOCATION = './data/nyc-zip-code-tabulation-areas-polygons.geojson'
US_STATES_JSON_FILE_LOCATION = './data/gz_2010_us_040_00_20m.json'

STATES_TO_DISPLAY = {'New York', 'New Jersey', 'Massachusetts', 'Connecticut', 'Rhode Island'}

######################
# Data Preprocessing #
######################

def load_borough_geojson() -> dict:
    with open(BOROUGH_GEOJSON_FILE_LOCATION, 'r') as f_borough:
        borough_geojson = json.loads(f_borough.read())
    return borough_geojson

def load_zip_code_geojson() -> dict:
    with open(ZIP_CODE_GEOJSON_FILE_LOCATION, 'r') as f_zip_code:
        zip_code_geojson = json.loads(f_zip_code.read())
    for feature in zip_code_geojson['features']:
        feature['properties']['Zip_Code'] = int(feature['properties']['postalCode'])
    return zip_code_geojson

def load_us_states_geojson() -> dict:
    with open(US_STATES_JSON_FILE_LOCATION, 'r') as f_states:
        us_states_geojson = json.loads(f_states.read())
    us_states_geojson['features'] = [feature for feature in us_states_geojson['features'] if feature['properties']['STATE'] in STATES_TO_DISPLAY]
    return us_states_geojson

def _guess_boroughs_and_zip_codes(df: pd.DataFrame, borough_geojson: dict, zip_code_geojson: dict) -> pd.DataFrame:
    '''Destructive.'''
    manager = mp.Manager()
    borough_to_polygons = manager.dict()
    assert {feature['geometry']['type'] for feature in borough_geojson['features']} == {'MultiPolygon'}
    for feature in borough_geojson['features']:
        borough_name = feature['properties']['boro_name'].lower()
        polygon_coordinate_lists = map(lambda coordinate_list: coordinate_list[0], feature['geometry']['coordinates'])
        polygons = eager_map(Polygon, polygon_coordinate_lists)
        borough_to_polygons[borough_name] = polygons
    def _guess_borough(row: pd.Series) -> Union[str, float]:
        point = Point(row['LONGITUDE'], row['LATITUDE'])
        nearest_borough = None
        nearest_borough_dist = float('inf')
        for borough, polygons in borough_to_polygons.items():
            for polygon in polygons:
                if point.within(polygon):
                    return borough
                borough_dist = polygon.exterior.distance(point)
                if borough_dist < nearest_borough_dist:
                    nearest_borough_dist = borough_dist
                    nearest_borough = borough
        assert isinstance(nearest_borough, str)
        nearest_borough = nearest_borough.lower() if nearest_borough_dist < 0.01 else np.nan
        return nearest_borough
    zip_code_to_polygon = mp.Manager().dict()
    assert {feature['geometry']['type'] for feature in zip_code_geojson['features']} == {'Polygon'}
    for feature in zip_code_geojson['features']:
        zip_code = feature['properties']['Zip_Code']
        polygon_coordinate_list = feature['geometry']['coordinates'][0]
        polygon = Polygon(polygon_coordinate_list)
        assert polygon.area != 0.0
        zip_code_to_polygon[zip_code] = polygon
    def _guess_zip_code(row: pd.Series) -> Union[int, float]:
        point = Point(row['LONGITUDE'], row['LATITUDE'])
        nearest_zip_code = None
        nearest_zip_code_dist = float('inf')
        for zip_code, polygon in zip_code_to_polygon.items():
            if point.within(polygon):
                return zip_code
            zip_code_dist = polygon.exterior.distance(point)
            assert zip_code_dist != 0.0
            if zip_code_dist < nearest_zip_code_dist:
                nearest_zip_code_dist = zip_code_dist
                nearest_zip_code = zip_code
        assert isinstance(nearest_zip_code, int)
        nearest_zip_code = nearest_zip_code if nearest_zip_code_dist < 0.05 else np.nan
        import random # @todo remove this
        if random.randint(1,10) == 1:
            print(f"row.name {repr(row.name)}")
        return nearest_zip_code
    # @todo flip these
    _is_non_numeric_string = lambda zip_code: isinstance(zip_code, str) and not str.isnumeric(zip_code)
    missing_zip_code_df = df[df['ZIP CODE'].isnull() | df['ZIP CODE'].parallel_map(_is_non_numeric_string)]
    df.loc[missing_zip_code_df.index, 'ZIP CODE'] = missing_zip_code_df.parallel_apply(_guess_zip_code, axis=1)
    breakpoint()
    df.loc[df[df['BOROUGH'].isnull()].index, 'BOROUGH'] = df.loc[df['BOROUGH'].isnull()].parallel_apply(_guess_borough, axis=1)
    breakpoint()
    return df

def load_crash_df(borough_geojson: dict, zip_code_geojson: dict) -> pd.DataFrame:
    with warnings_suppressed():
        df = pd.read_csv(CRASH_DATA_CSV_FILE_LOCATION)
    df.drop(df[df['LATITUDE'].isnull()].index, inplace=True)
    df.drop(df[df['LONGITUDE'].isnull()].index, inplace=True)
    df.drop(df[df['LONGITUDE'].eq(0.0) & df['LATITUDE'].eq(0.0)].index, inplace=True)
    df['CRASH HOUR'] = df['CRASH TIME'].map(lambda crash_time_string: int(crash_time_string.split(':')[0]))
    df = _guess_boroughs_and_zip_codes(df, borough_geojson, zip_code_geojson)
    df['BOROUGH'] = df['BOROUGH'].map(str.lower)
    df.astype({'ZIP CODE': int}, copy=False)
    return df

##########
# Driver #
##########

def _sanity_check_data(df: pd.DataFrame, borough_geojson: dict, zip_code_geojson: dict, us_states_geojson: dict) -> None:
    if __debug__:
        df_boroughs = {e for e in df['BOROUGH'].unique() if isinstance(e, str)}
        borough_geojson_boroughs = {feature['properties']['boro_name'].lower() for feature in borough_geojson['features']}
        assert df_boroughs == borough_geojson_boroughs
        df_zip_codes = set(df['ZIP CODE'].unique().tolist())
        zip_code_geojson_zip_codes = {feature['properties']['Zip_Code'] for feature in borough_geojson['features']}
        assert df_zip_codes == zip_code_geojson_zip_codes
        assert {feature['properties']['STATE'] for feature in us_states_geojson['features']} == STATES_TO_DISPLAY
    return 

@debug_on_error
def main() -> None:
    borough_geojson = load_borough_geojson()
    zip_code_geojson = load_zip_code_geojson()
    us_states_geojson = load_us_states_geojson()
    df = load_crash_df(borough_geojson, zip_code_geojson)
    _sanity_check_data(df, borough_geojson, zip_code_geojson, us_states_geojson)
    return

if __name__ == '__main__':
    main()
