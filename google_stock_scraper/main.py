
'''

'''

# TODO fill in doc string

###########
# Imports #
###########

import os
import subprocess
import time
import tempfile
import sqlite3
import bs4
import requests
from more_itertools import distribute
from  typing import Set

from misc_utilities import *

# TODO make sure these imports are used

###########
# Globals #
###########

NUMBER_OF_SUBPROCESSES = 8

SUBPROCESS_COMMAND_TEMPLATE = 'python3 scrape.py -output {stock_data_db_file} -ticker-symbol-file {ticker_symbol_file}'

ALL_TICKER_SYMBOLS_URL = 'https://stockanalysis.com/stocks/'

STOCK_DATA_DB_FILE = './stock_data.db'

#########################
# Gather Ticker Symbols #
#########################

def gather_ticker_symbols() -> List[str]:
    response = requests.get(ALL_TICKER_SYMBOLS_URL)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    ticker_links = soup.select('main.site-main article.page.type-page.status-publish div.inside-article div.entry-content ul.no-spacing li a')
    ticker_symbols = [ticker_link.text.split(' - ')[0] for ticker_link in ticker_links]    
    return ticker_symbols

##########
# Driver #
##########

@debug_on_error
def poll_scraping_subprocess(process: subprocess.Popen, pids: Set[int]) -> None:
    last_message = ''
    while process.poll() is None:
        with open(LOGGER_OUTPUT_FILE, 'r') as f:
            lines = f.readlines()
        pid_to_message_line = {}
        lines = reversed(lines)
        lines = filter(lambda line: '] [' in line, lines)
        mean_speed = 0
        for line in lines:
            line_pid = int(line.split(' - threadid: ')[0].split(' - pid: ')[1])
            if line_pid in pids and line_pid not in pid_to_message_line:
                pid_progress, pid_speed, _ = line.split('] [')
                pid_progress = pid_progress.split('[')[1]
                pid_speed = float(pid_speed.split(' ')[-2])
                mean_speed += pid_speed
                pid_to_message_line[line_pid] = f'[{pid_progress}] Process {line_pid:5} is operating at {pid_speed} s/iter.'
                if len(pid_to_message_line) == NUMBER_OF_SUBPROCESSES:
                    break
        if len(pid_to_message_line) > 0:
            mean_speed /= len(pid_to_message_line)
            breaker_text = max(map(len, pid_to_message_line.values())) * '='
            message = '\n'.join([pid_to_message_line[pid] for pid in sorted(pid_to_message_line.keys())])
            message += '\n' + f'Mean process speed: {mean_speed:.2} s/iter.'
            message += '\n' + f'Total process speed: {mean_speed/len(pids):.2} s/iter.'
            message = '\n' + breaker_text + '\n' + message + '\n' + breaker_text
            if message != last_message:
                print(message)
                last_message = message
    process.wait()
    assert process.poll() == 0
    return
    
if __name__ == '__main__':
    connection = sqlite3.connect(STOCK_DATA_DB_FILE)
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS stocks(date timestamp, ticker_symbol text, price real)')
    
    ticker_symbols = gather_ticker_symbols()
    LOGGER.info(f'{len(ticker_symbols)} ticker symbols gathered.')

    with tempfile.TemporaryDirectory() as temporary_directory:
        
        db_file_to_process = {}
        
        for subprocess_id, ticker_symbol_batch in enumerate(distribute(NUMBER_OF_SUBPROCESSES, ticker_symbols)):
            stock_data_db_file = os.path.join(temporary_directory, f'stock_data_{subprocess_id}.db')
            ticker_symbol_file = os.path.join(temporary_directory, f'ticker_symbols_{subprocess_id}.txt')
            with open(ticker_symbol_file, 'w') as f:
                f.write('\n'.join(ticker_symbol_batch))
            command = SUBPROCESS_COMMAND_TEMPLATE.format(
                stock_data_db_file=stock_data_db_file,
                ticker_symbol_file=ticker_symbol_file
            ).split()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            db_file_to_process[stock_data_db_file] = process

        pids = {process.pid for process in db_file_to_process.values()}
        for stock_data_db_file, process in db_file_to_process.items():
            poll_scraping_subprocess(process, pids)
            
            subprocess_connection = sqlite3.connect(stock_data_db_file)
            subprocess_cursor = subprocess_connection.cursor()
            subprocess_cursor.execute('SELECT date, ticker_symbol, price FROM stocks')
            rows = subprocess_cursor.fetchall()

            cursor.executemany('INSERT INTO stocks VALUES(?,?,?);', rows);
            connection.commit();

    connection.close()
