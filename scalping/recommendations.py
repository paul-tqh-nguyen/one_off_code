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

from misc_utilities import *

import bokeh.plotting
import bokeh.models
 
# @todo make sure these are used

###########
# Globals #
###########

RELLEVANT_TICKER_SYMBOLS = [
    'FB',
    'AMZN',
    # 'AAPL',
    # 'NFLX',
    # 'GOOGL',
    # 'TSLA',
    # 'CMG',
]

NUMBER_OF_DAYS = 5

ENDPOINT_TEMPLATE = 'https://api.tiingo.com/iex/{ticker_symbol}/prices?startDate={start_date_string}&endDate={end_date_string}&resampleFreq=5min'

TOKEN = os.environ.get('TOKEN')
assert TOKEN, 'No authentication token specified.'

DOCS_DIR = './docs'

OUTPUT_DIR = './docs/output'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdirs(OUTPUT_DIR)

#############################
# Domain-Specific Utilities #
#############################

def get_nth_previous_business_day(n: int) -> datetime.date:
    today = datetime.date.today()
    nth_previous_business_day = today - pd.tseries.offsets.BDay(n)
    nth_previous_business_day = nth_previous_business_day.date()
    return nth_previous_business_day

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    start_date_string = get_nth_previous_business_day(NUMBER_OF_DAYS).strftime("%Y-%m-%d")
    end_date_string = datetime.date.today().strftime("%Y-%m-%d")
    ticker_data = recursive_defaultdict()
    for ticker_symbol in RELLEVANT_TICKER_SYMBOLS:
        endpoint_url = ENDPOINT_TEMPLATE.format(ticker_symbol=ticker_symbol, start_date_string=start_date_string, end_date_string=end_date_string)
        response = requests.get(endpoint_url, headers={'Content-Type': 'application/json', 'Authorization' : f'Token {TOKEN}'})
        assert response.status_code == 200
        df = pd.DataFrame(response.json())
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert('US/Central').dt.tz_localize(None)
        df['truncated_date'] = df['date'].map(datetime.datetime.date)
        df['time_of_day_string'] = df['date'].map(lambda date: date.strftime('%H:%M'))
        for current_date, group in df.groupby('truncated_date'):
            p = bokeh.plotting.figure(
                title=f'{ticker_symbol} Stock Prices for {current_date}',
                x_axis_type='datetime',
                y_axis_type='linear',
                x_axis_label = 'Time',
                y_axis_label = 'Stock Price',
                plot_height=400,
                plot_width=800,
                x_range=(group.date.min(), group.date.max()),
                tools='hover, crosshair',
            )
            p.toolbar.logo = None
            p.toolbar_location = None
            source = bokeh.models.ColumnDataSource(data=group)
            p.line(x='date', y='open', source=source, line_width=3)
            hovertool = p.select_one(bokeh.models.HoverTool)
            hovertool.tooltips = [
                ('Stock Price', '@open{$0,0.00}'),
                ('Time of Day', '@time_of_day_string'),
                ('Ticker Symbol', ticker_symbol),
            ]
            p.xaxis.formatter = bokeh.models.DatetimeTickFormatter(
                hours=['%I:%M %p'],
            )
            hovertool.mode = 'vline'
            output_file_location = os.path.join(OUTPUT_DIR, f'{ticker_symbol}_{current_date}_plot.html')
            bokeh.plotting.output_file(output_file_location, title='{ticker_symbol} Stock Prices {current_date}', mode='inline')
            bokeh.plotting.save(p, title='{current_date} {ticker_symbol} Stock Prices')
            ticker_data[ticker_symbol][current_date.isoformat()] = {'html_file': os.path.relpath(output_file_location, DOCS_DIR)} # @todo add more data to this dict
    return

if __name__ == '__main__':
    main()
