
########################
# RH Browser Utilities #
########################

RH_BROWSER = None

TRACKED_TICKER_SYMBOL_TO_PAGE: Dict[str, pyppeteer.page.Page] = dict()

async def _initialize_rh_browser() -> pyppeteer.browser.Browser:
    browser = await launch_browser(
        headless=True,
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False
    )
    return browser

@event_loop_func
def initialize_rh_browser() -> pyppeteer.browser.Browser:
    global RH_BROWSER
    assert RH_BROWSER is None, f'RH browser already initialized.'
    browser = run_awaitable(_initialize_rh_browser())
    RH_BROWSER = browser
    return browser

@event_loop_func
def close_rh_browser() -> None:
    global RH_BROWSER
    assert RH_BROWSER is not None, f'RH browser already closed.'
    run_awaitable(RH_BROWSER._closeCallback())
    RH_BROWSER = None
    return 

########################
# RH Scraper Utilities #
########################

RH_SCRAPER_TASK = None

async def open_pages_for_ticker_symbols(ticker_symbols: List[str]) -> None:
    global TRACKED_TICKER_SYMBOL_TO_PAGE
    ticker_symbols = eager_map(str.upper, ticker_symbols)
    current_pages = await RH_BROWSER.pages()
    
    # Add new pages
    for _ in range(len(ticker_symbols) - len(current_pages)):
        await RH_BROWSER.newPage()
    
    # Close unneccessary pages
    for index in range(len(current_pages) - len(ticker_symbols)):
        await current_pages[index].close()
    
    more_itertools.consume((TRACKED_TICKER_SYMBOL_TO_PAGE.popitem() for _ in range(len(TRACKED_TICKER_SYMBOL_TO_PAGE))))
    assert len(TRACKED_TICKER_SYMBOL_TO_PAGE) == 0
    
    current_pages = await RH_BROWSER.pages()
    assert len(current_pages) == len(ticker_symbols)
    
    page_opening_tasks = []
    for page, ticker_symbol in zip(current_pages, ticker_symbols):
        url = f'https://robinhood.com/stocks/{ticker_symbol}'
        page_opening_task = EVENT_LOOP.create_task(page.goto(url))
        page_opening_tasks.append(page_opening_task)
        TRACKED_TICKER_SYMBOL_TO_PAGE[ticker_symbol] = page
    for page_opening_task in page_opening_tasks:
        await page_opening_task
    
    return

async def scrape_ticker_symbols() -> None:
    try:
        while True:
            rows = []
            
            # Trigger page updates via mouse movements first
            animation_triggering_time = time.time()
            for ticker_symbol, page in TRACKED_TICKER_SYMBOL_TO_PAGE.items():
                assert page.url.split('/')[-1].lower() == ticker_symbol.lower()
                await page.bringToFront()
                
                svg = await page.get_sole_element('body main.app main.main-container div.row section[data-testid="ChartSection"] svg:not([role="img"])')
                top, left, width, height = await page.evaluate(
                    '(element) => {'
                    '    const { top, left, width, height } = element.getBoundingClientRect();'
                    '    return [top, left, width, height];'
                    '}', svg)
    
                for _ in range(10):
                    y = top + (top + height) * random.random()
                    x = left + (left+width) * random.random()
                    await page.mouse.move(x, y)
                await page.mouse.move(1, 1)

            sleep_time = time.time() - animation_triggering_time
            sleep_time = 1-sleep_time
            sleep_time = max(sleep_time, 0)
            await asyncio.sleep(sleep_time) # let animations settle for all tabs
            
            # Perform actual scraping
            for ticker_symbol, page in TRACKED_TICKER_SYMBOL_TO_PAGE.items():
                await page.bringToFront()
                
                price_spans = await page.get_elements('body main.app main.main-container div.row section[data-testid="ChartSection"] header')
                price_span_string = await page.evaluate('(element) => element.innerText', price_spans[0])
                price_span_string = price_span_string.split('\n')[0]
                assert price_span_string.startswith('$')
                price = float(price_span_string.replace('$', '').replace(',', ''))
                
                date_time = get_local_datetime()
                row = (date_time, ticker_symbol, price)
                rows.append(row)

            with DB_INFO.db_access() as (db_connection, db_cursor):
                db_cursor.executemany('INSERT INTO stocks VALUES(?,?,?)', rows)
                db_connection.commit()
            
    except asyncio.CancelledError:
        pass
    return

@event_loop_func
def track_ticker_symbols(*ticker_symbols: List[str]) -> None:
    global RH_BROWSER
    global RH_SCRAPER_TASK
    assert RH_BROWSER is not None, f'RH browser not initialized.'
    if RH_SCRAPER_TASK is not None:
        RH_SCRAPER_TASK.cancel()
    run_awaitable(open_pages_for_ticker_symbols(ticker_symbols))
    RH_SCRAPER_TASK = enqueue_awaitable(scrape_ticker_symbols())
    return
