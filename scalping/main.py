#!/usr/bin/python3 -OO

'''
'''

# @todo update doc string

###########
# Imports #
###########

import os
import requests
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.cm

from misc_utilities import *

import bokeh.plotting
import bokeh.models
 
# @todo make sure these are used

###########
# Globals #
###########

RELEVANT_TICKER_SYMBOLS = [
    'FB',
    'AAPL',
    'AMZN',
    'NFLX',
    'GOOGL',
    'TSLA',
    'CMG',
    'ULTA',
    'PG',
    'SPOT',
    'NKE',
    'SBUX',
    'TGT',
]

NUMBER_OF_DAYS = 5

ENDPOINT_TEMPLATE = 'https://api.tiingo.com/iex/{ticker_symbol}/prices?startDate={start_date_string}&endDate={end_date_string}&resampleFreq=5min'

TOKEN = os.environ.get('TOKEN')
assert TOKEN, 'No authentication token specified.'

TODAY = datetime.date.today()

DOCS_DIR = './docs'

OUTPUT_DIR = './docs/output'
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_JSON_FILE_LOCATION = os.path.join(OUTPUT_DIR, 'output_summary.json')

COLOR_MAP = matplotlib.cm.get_cmap('Dark2')

#############################
# Domain-Specific Utilities #
#############################

def get_nth_previous_business_day(n: int) -> datetime.date:
    reference_date = TODAY
    nth_previous_business_day = reference_date - pd.tseries.offsets.BDay(n)
    nth_previous_business_day = nth_previous_business_day.date()
    return nth_previous_business_day

#################
# Visualization #
#################

def create_day_chart(current_date: pd.Timestamp, ticker_symbol: str, group: pd.DataFrame, line_color: np.ndarray, output_file_location: str) -> None:
    p = bokeh.plotting.figure(
        title=f'{ticker_symbol} Stock Prices for {current_date}',
        x_axis_type='datetime',
        y_axis_type='linear',
        x_axis_label = 'Time',
        y_axis_label = 'Stock Price',
        plot_height=250,
        plot_width=400,
        x_range=(group.date.min(), group.date.max()),
        tools='',
    )
    p.toolbar.logo = None
    p.toolbar_location = None
    source = bokeh.models.ColumnDataSource(data=group)
    p.line(x='date', y='open', source=source, line_width=1, line_color=line_color)
    hovertool = bokeh.models.HoverTool(mode='vline')
    hovertool.tooltips = [
        ('Stock Price', '@open{$0,0.00}'),
        ('Time of Day', '@time_of_day_string'),
    ]
    p.add_tools(hovertool)
    crosshairtool = bokeh.models.CrosshairTool(dimensions='height')
    p.add_tools(crosshairtool)
    p.xaxis.formatter = bokeh.models.DatetimeTickFormatter(hours=['%I:%M %p'])
    bokeh.plotting.output_file(output_file_location, title=p.title.text, mode='inline')
    bokeh.plotting.save(p, title='{current_date} {ticker_symbol} Stock Prices')
    return

def create_combined_days_chart(df: pd.DataFrame, ticker_symbol: str, output_file_location: str) -> None:
    start_date_string = df[df.date == df.date.min()].truncated_date.values.item()
    end_date_string = df[df.date == df.date.max()].truncated_date.values.item()
    p = bokeh.plotting.figure(
        title=f'{ticker_symbol} Stock Price changes from {start_date_string} to {end_date_string}',
        x_axis_type='datetime',
        y_axis_type='linear',
        x_axis_label = 'Time',
        y_axis_label = 'Percent of Open Price',
        plot_height=250,
        plot_width=400,
        x_range=(df.time_of_day.min(), df.time_of_day.max()),
    )
    p.toolbar.logo = None
    p.toolbar_location = None
    hovertool = bokeh.models.HoverTool(mode='mouse')
    hovertool.tooltips = [
        ('Percent of Open Price', '@percent_of_open_price'),
        ('Time of Day', '@time_of_day_string'),
        ('Date','@date_string'),
    ]
    p.add_tools(hovertool)
    colors = eager_map(tuple, (COLOR_MAP(np.linspace(0, 1, len(df.truncated_date.unique())))[:,:3]*255).tolist())
    for group_index,  (current_date, group) in enumerate(df.groupby('truncated_date')):
        source = bokeh.models.ColumnDataSource(data=group)
        p.line(x='time_of_day', y='percent_of_open_price', source=source, line_width=1, line_color=bokeh.colors.RGB(*colors[group_index]))
    crosshairtool = bokeh.models.CrosshairTool(dimensions='height')
    p.add_tools(crosshairtool)
    p.xaxis.formatter = bokeh.models.DatetimeTickFormatter(hours=['%I:%M %p'])
    bokeh.plotting.output_file(output_file_location, title=p.title.text, mode='inline')
    bokeh.plotting.save(p, title='{current_date} {ticker_symbol} Stock Prices')
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    start_date_string = get_nth_previous_business_day(NUMBER_OF_DAYS).strftime("%Y-%m-%d")
    end_date_string = TODAY.strftime("%Y-%m-%d")
    ticker_data = recursive_defaultdict()
    for ticker_symbol in RELEVANT_TICKER_SYMBOLS:
        endpoint_url = ENDPOINT_TEMPLATE.format(ticker_symbol=ticker_symbol, start_date_string=start_date_string, end_date_string=end_date_string)
        response = requests.get(endpoint_url, headers={'Content-Type': 'application/json', 'Authorization' : f'Token {TOKEN}'})
        assert response.status_code == 200
        df = pd.DataFrame(response.json())
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert('US/Central').dt.tz_localize(None)
        df.sort_values('date', inplace=True)
        df['date_string'] = df['date'].map(lambda date: date.strftime('%Y-%m-%d'))
        df['truncated_date'] = df['date'].map(datetime.datetime.date)
        df['time_of_day'] = df['date'].dt.time
        df['time_of_day_string'] = df['date'].map(lambda date: date.strftime('%H:%M'))
        df['percent_of_open_price'] = (df.groupby('truncated_date').apply(lambda group: group.open/group[group.date==group.date.min()].open.values.item()).droplevel(0)-1)*100
        colors = eager_map(tuple, (COLOR_MAP(np.linspace(0, 1, len(df.truncated_date.unique())))[:,:3]*255).tolist())
        for group_index, (current_date, group) in enumerate(df.groupby('truncated_date')):
            date_output_file_location = os.path.join(OUTPUT_DIR, f'{ticker_symbol}_{current_date}_plot.html')
            line_color = bokeh.colors.RGB(*colors[group_index])
            create_day_chart(current_date, ticker_symbol, group, line_color, date_output_file_location)
            ticker_data[ticker_symbol]['date_data'][current_date.isoformat()] = {
                'html_file': os.path.relpath(date_output_file_location, DOCS_DIR),
                'opening_price': group[group.date == group.date.min()].open.values.item(),
                'closing_price': group[group.date == group.date.max()].close.values.item(),
                'min_price': group.open.min(),
                'max_price': group.open.max()
            }
        combined_days_output_file_location = os.path.join(OUTPUT_DIR, f'{ticker_symbol}_combined_days_plot.html')
        create_combined_days_chart(df, ticker_symbol, combined_days_output_file_location)
        ticker_data[ticker_symbol]['combined_date_html_file'] = os.path.relpath(combined_days_output_file_location, DOCS_DIR)
    with open(OUTPUT_JSON_FILE_LOCATION, 'w') as file_handle:
        json.dump(ticker_data, file_handle, indent=4)
    return

if __name__ == '__main__':
    main()
