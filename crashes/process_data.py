'#!/usr/bin/python3 -OO' # @todo make this the default

'''
'''
# @todo update doc string

###########
# Imports #
###########

import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import multiprocessing.managers
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

OUTPUT_JSON_FILE_LOCATION = './docs/processed_data.json'

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
    us_states_geojson['features'] = [feature for feature in us_states_geojson['features'] if feature['properties']['NAME'] in STATES_TO_DISPLAY]
    return us_states_geojson

def _guess_boroughs_and_zip_codes(df: pd.DataFrame, borough_geojson: dict, zip_code_geojson: dict) -> pd.DataFrame:
    '''Destructive.'''
    with mp.Manager() as manager:
        zip_code_to_polygon = manager.dict()
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
        _is_non_numeric_string = lambda zip_code: isinstance(zip_code, str) and not str.isnumeric(zip_code)
        missing_zip_code_df = df[df['ZIP CODE'].isnull() | df['ZIP CODE'].parallel_map(_is_non_numeric_string)]
        df.loc[missing_zip_code_df.index, 'ZIP CODE'] = missing_zip_code_df.parallel_apply(_guess_zip_code, axis=1)
        df.loc[df[df['BOROUGH'].isnull()].index, 'BOROUGH'] = df[df['BOROUGH'].isnull()].parallel_apply(_guess_borough, axis=1)
    return df

def load_crash_df(borough_geojson: dict, zip_code_geojson: dict) -> pd.DataFrame:
    print('Loading crash data.')
    with warnings_suppressed():
        df = pd.read_csv(CRASH_DATA_CSV_FILE_LOCATION, parse_dates=['CRASH DATE'])
    df.drop(df[df['LATITUDE'].isnull()].index, inplace=True)
    df.drop(df[df['LONGITUDE'].isnull()].index, inplace=True)
    df.drop(df[df['LONGITUDE'].eq(0.0) & df['LATITUDE'].eq(0.0)].index, inplace=True)
    df['CRASH HOUR'] = df['CRASH TIME'].parallel_map(lambda crash_time_string: int(crash_time_string.split(':')[0]))
    df = df.sample(200) # @todo remove this
    print('Adding missing borough and zip code data.')
    df = _guess_boroughs_and_zip_codes(df, borough_geojson, zip_code_geojson)
    df.drop(df[df['BOROUGH'].isnull()].index, inplace=True)
    df.drop(df[df['ZIP CODE'].isnull()].index, inplace=True)
    df['ZIP CODE'] = df['ZIP CODE'].map(int)
    df['BOROUGH'] = df['BOROUGH'].parallel_map(str.lower)
    df.astype({'ZIP CODE': int}, copy=False)
    return df

def convert_proxy_data_structures_to_normal_datastructures(proxy: Union[mp.managers.DictProxy, mp.managers.ListProxy]) -> Union[list, dict]:
    if isinstance(proxy, mp.managers.DictProxy):
        result = dict()
        for key, value in proxy.items():
            if isinstance(value, mp.managers.DictProxy) or isinstance(value, mp.managers.ListProxy):
                result[key] = convert_proxy_data_structures_to_normal_datastructures(value)
            else:
                result[key] = value
    elif isinstance(proxy, mp.managers.ListProxy):
        result = []
        for item in proxy:
            if isinstance(item, mp.managers.DictProxy) or isinstance(item, mp.managers.ListProxy):
                result.append(convert_proxy_data_structures_to_normal_datastructures(item))
            else:
                result.append(item)
        assert len(result) > 0
    else:
        raise ValueError(f"Cannot extract non-proxy type from {proxy}.")
    return result

def generate_output_dict(df: pd.DataFrame, borough_geojson: dict, zip_code_geojson: dict, us_states_geojson: dict) -> dict:
    '''output_dict has indexing of date -> hour -> borough -> zip code'''
    print('Generating output dictionary.')
    output_dict = dict()
    with mp.Manager() as manager:
        crash_data_dict = manager.dict()

        def _note_zip_code_rows(group: pd.DataFrame, dict_for_borough: mp.managers.DictProxy) -> None:
            zip_code: int = only_one(group['ZIP CODE'].unique())
            dict_for_borough[zip_code] = list(group.to_dict(orient='index').values())
            return
        
        def _note_borough_rows(group: pd.DataFrame, dict_for_hour: mp.managers.DictProxy) -> None:
            borough = only_one(group['BOROUGH'].unique())
            dict_for_hour[borough] = manager.dict()
            dict_for_borough = dict_for_hour[borough]
            group.groupby('ZIP CODE').apply(lambda zip_code_group: _note_zip_code_rows(zip_code_group, dict_for_borough))
            return 
        
        def _note_hour_rows(group: pd.DataFrame, array_for_date: mp.managers.ListProxy) -> None:
            hour = only_one(group['CRASH HOUR'].unique())
            dict_for_hour = array_for_date[hour]
            group.groupby('BOROUGH').apply(lambda borough_group: _note_borough_rows(borough_group, dict_for_hour))
            return 
        
        def _note_date_rows(group: pd.DataFrame) -> None:
            date_string = pd.to_datetime(only_one(group['CRASH DATE'].unique())).isoformat()
            output_dict[date_string] = manager.list([manager.dict() for _ in range(24)])
            array_for_date = output_dict[date_string]
            group.groupby('CRASH HOUR').apply(lambda hour_group: _note_hour_rows(hour_group, array_for_date))
            return
        
        df.groupby('CRASH DATE').parallel_apply(_note_date_rows)
        crash_data_dict = convert_proxy_data_structures_to_normal_datastructures(output_dict)
    output_dict['crash_data'] = crash_data_dict
    output_dict['borough_data'] = borough_geojson
    output_dict['zip_code_data'] = zip_code_geojson
    output_dict['states_data'] = us_states_geojson
    return output_dict

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

@debug_on_error
def main() -> None:
    borough_geojson = load_borough_geojson()
    zip_code_geojson = load_zip_code_geojson()
    us_states_geojson = load_us_states_geojson()
    df = load_crash_df(borough_geojson, zip_code_geojson)
    _sanity_check_data(df, borough_geojson, zip_code_geojson, us_states_geojson)
    output_dict = generate_output_dict(df, borough_geojson, zip_code_geojson, us_states_geojson)
    with open(OUTPUT_JSON_FILE_LOCATION, 'w') as file_handle:
        json.dump(output_dict, file_handle, indent=4)
    breakpoint()
    return

if __name__ == '__main__':
    main()
