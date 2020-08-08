#!/usr/bin/python3 -OO

'''
'''
# @todo update doc string

###########
# Imports #
###########

import json
import datetime
import numpy as np
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

# https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95 
CRASH_DATA_CSV_FILE_LOCATION = './data/Motor_Vehicle_Collisions_-_Crashes.csv'

# https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm
BOROUGH_GEOJSON_FILE_LOCATION = './data/Borough Boundaries.geojson'

# https://github.com/fedhere/PUI2015_EC/blob/master/mam1612_EC/nyc-zip-code-tabulation-areas-polygons.geojson
NYC_ZIP_CODE_GEOJSON_FILE_LOCATION = './data/nyc-zip-code-tabulation-areas-polygons.geojson'

# https://eric.clst.org/tech/usgeojson/
US_STATES_JSON_FILE_LOCATION = './data/gz_2010_us_040_00_20m.json'

OUTPUT_BOROUGH_JSON_FILE_LOCATION = './docs/borough.json'
OUTPUT_ZIP_CODE_JSON_FILE_LOCATION = './docs/zip_code.json'
OUTPUT_STATES_JSON_FILE_LOCATION = './docs/states.json'
OUTPUT_CRASH_JSON_FILE_LOCATION = './docs/crash.json'

STATES_TO_DISPLAY = {'New York', 'New Jersey', 'Massachusetts', 'Connecticut', 'Rhode Island'}

######################
# Data Preprocessing #
######################

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        else:
            return super(CustomEncoder, self).default(obj)

def load_borough_geojson() -> dict:
    with open(BOROUGH_GEOJSON_FILE_LOCATION, 'r') as f_borough:
        borough_geojson = json.loads(f_borough.read())
    return borough_geojson

def load_zip_code_geojson() -> dict:
    with open(NYC_ZIP_CODE_GEOJSON_FILE_LOCATION, 'r') as f_zip_code:
        zip_code_geojson = json.loads(f_zip_code.read())
    for feature in zip_code_geojson['features']:
        feature['properties']['Zip_Code'] = int(feature['properties']['postalCode'])
    return zip_code_geojson

def load_us_states_geojson() -> dict:
    with open(US_STATES_JSON_FILE_LOCATION, 'r') as f_states:
        us_states_geojson = json.loads(f_states.read())
    us_states_geojson['features'] = [feature for feature in us_states_geojson['features'] if feature['properties']['NAME'] in STATES_TO_DISPLAY]
    return us_states_geojson

def _guess_boroughs_and_zip_codes(df: pd.DataFrame, borough_geojson: dict, zip_code_geojson: dict) -> pd.DataFrame:
    '''Destructive.'''
    zip_code_to_polygon = {}
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
        return nearest_zip_code
    borough_to_polygons = {}
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
    _is_non_numeric_string = lambda zip_code: isinstance(zip_code, str) and not str.isnumeric(zip_code)
    missing_zip_code_indexer = df['ZIP CODE'].isnull() | df['ZIP CODE'].parallel_map(_is_non_numeric_string)
    print(f'Numbers of rows with missing ZIP codes before ZIP code estimation: {missing_zip_code_indexer.sum():,} ({round(missing_zip_code_indexer.sum()/len(df)*100, 2)}%)')
    df.loc[missing_zip_code_indexer, 'ZIP CODE'] = df[missing_zip_code_indexer].parallel_apply(_guess_zip_code, axis=1)
    missing_zip_code_indexer = df['ZIP CODE'].isnull() | df['ZIP CODE'].parallel_map(_is_non_numeric_string)
    print(f'Numbers of rows with missing ZIP codes after ZIP code estimation: {missing_zip_code_indexer.sum():,} ({round(missing_zip_code_indexer.sum()/len(df)*100, 2)}%)')
    missing_borough_indexer = df['BOROUGH'].isnull()
    print(f'Numbers of rows with missing boroughs before borough estimation: {missing_borough_indexer.sum():,} ({round(missing_borough_indexer.sum()/len(df)*100, 2)}%)')
    df.loc[missing_borough_indexer, 'BOROUGH'] = df[missing_borough_indexer].parallel_apply(_guess_borough, axis=1)
    missing_borough_indexer = df['BOROUGH'].isnull()
    print(f'Numbers of rows with missing boroughs after borough estimation: {missing_borough_indexer.sum():,} ({round(missing_borough_indexer.sum()/len(df)*100, 2)}%)')
    return df

def load_crash_df(borough_geojson: dict, zip_code_geojson: dict) -> pd.DataFrame:
    print('Loading crash data.')
    with warnings_suppressed():
        df = pd.read_csv(CRASH_DATA_CSV_FILE_LOCATION, parse_dates=['CRASH DATE'])
    print(f'Raw data size: {len(df):,}')
    df.drop(df[df['LATITUDE'].isnull()].index, inplace=True)
    df.drop(df[df['LONGITUDE'].isnull()].index, inplace=True)
    df.drop(df[df['LONGITUDE'].eq(0.0) & df['LATITUDE'].eq(0.0)].index, inplace=True)
    print(f'Data size after dropping NaNs: {len(df):,}')
    assert df['LONGITUDE'].isnull().sum() == df['LATITUDE'].isnull().sum() == 0
    df['CRASH HOUR'] = df['CRASH TIME'].parallel_map(lambda crash_time_string: int(crash_time_string.split(':')[0]))
    print('Adding missing borough and zip code data.')
    zip_code_geojson_zip_codes = {feature['properties']['Zip_Code'] for feature in zip_code_geojson['features']}
    df.loc[(~df['ZIP CODE'].isin(zip_code_geojson_zip_codes)).index, 'ZIP CODE'] = np.nan
    df = _guess_boroughs_and_zip_codes(df, borough_geojson, zip_code_geojson)
    df.drop(df[df['ZIP CODE'].isnull()].index, inplace=True)
    df.drop(df[df['BOROUGH'].isnull()].index, inplace=True)
    df = df.astype({'ZIP CODE': int}, copy=False)
    df['BOROUGH'] = df['BOROUGH'].parallel_map(str.lower)
    print(f'Final number of crashes: {len(df):,}')
    return df

def _note_date_group(date_to_date_group_pair: Tuple[pd.Timestamp, pd.DataFrame]) -> Tuple[str, list]:
    date, date_group = date_to_date_group_pair
    date_string = pd.to_datetime(only_one(date_group['CRASH DATE'].unique())).isoformat()
    array_for_date = [{} for _ in range(24)]
    for crash_hour, hour_group in date_group.groupby('CRASH HOUR'):
        assert isinstance(crash_hour, int)
        dict_for_hour = array_for_date[int(crash_hour)]
        for borough, borough_group in hour_group.groupby('BOROUGH'):
            dict_for_hour[borough] = {}
            dict_for_borough = dict_for_hour[borough]
            for zip_code, zip_code_group in borough_group.groupby('ZIP CODE'):
                dict_for_borough[zip_code] = list(zip_code_group.to_dict(orient='index').values())
    return (date_string, array_for_date)
    
def generate_crash_dict(df: pd.DataFrame) -> dict:
    '''output_dict has indexing of date -> hour -> borough -> zip code'''
    print('Generating crash dictionary.')
    crash_dict = dict(parallel_map(_note_date_group, df.groupby('CRASH DATE')))
    return crash_dict

##########
# Driver #
##########

def _sanity_check_data(df: pd.DataFrame, borough_geojson: dict, zip_code_geojson: dict, us_states_geojson: dict) -> None:
    if __debug__:
        df_boroughs = {e for e in df['BOROUGH'].unique() if isinstance(e, str)}
        borough_geojson_boroughs = {feature['properties']['boro_name'].lower() for feature in borough_geojson['features']}
        assert df_boroughs == borough_geojson_boroughs
        df_zip_codes = set(df['ZIP CODE'].unique().tolist())
        zip_code_geojson_zip_codes = {feature['properties']['Zip_Code'] for feature in zip_code_geojson['features']}
        assert df_zip_codes.issubset(zip_code_geojson_zip_codes)
        assert {feature['properties']['NAME'] for feature in us_states_geojson['features']} == STATES_TO_DISPLAY
    return

def write_dict_to_json_file(file_location: str, input_dict: dict) -> None:
    with open(file_location, 'w') as file_handle:
        json.dump(input_dict, file_handle, indent=4, cls=CustomEncoder)
    return 

@debug_on_error
def main() -> None:
    borough_geojson = load_borough_geojson()
    zip_code_geojson = load_zip_code_geojson()
    us_states_geojson = load_us_states_geojson()
    df = load_crash_df(borough_geojson, zip_code_geojson)
    _sanity_check_data(df, borough_geojson, zip_code_geojson, us_states_geojson)
    crash_dict = generate_crash_dict(df)
    write_dict_to_json_file(OUTPUT_BOROUGH_JSON_FILE_LOCATION, borough_geojson)
    write_dict_to_json_file(OUTPUT_ZIP_CODE_JSON_FILE_LOCATION, zip_code_geojson)
    write_dict_to_json_file(OUTPUT_STATES_JSON_FILE_LOCATION, us_states_geojson)
    write_dict_to_json_file(OUTPUT_CRASH_JSON_FILE_LOCATION, crash_dict)
    return

if __name__ == '__main__':
    main()
