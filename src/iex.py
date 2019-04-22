'''
Some Notes on pyEX api:
enpoint: /ref-data/symbols for symbols supported
https://api.iextrading.com/1.0/stock/aapl/chart/5y
this might be scrambled...
'''

# Pipeline
import requests
import re
from datetime import datetime

def make_doc_from_API(symbol, timeframe, summ_wanted = ['symbol', 'companyName', 'description', 'CEO', 'sector', 'tags']):
    '''NOTE: API May have already been sunsetted
    Using the symbol, and timeframe ['5y', '2y', '1y', 'ytd', '6m', '3m', '1m'] this function
    creates a document for insertion in to MongoDB. It utilizes the document oriented, schema-less ideas
    and typically allows for quicker read/write speeds as well as scalability and easy item changing. 
    Inputs: Symbol ('AAPL', 'aapl'), Timeframe (From Valid Choices)
    Returns: Document (Dictionary in JSON with Nested Dictionaries in 'data')
    '''
    valid_times = ['5y', '2y', '1y', 'ytd', '6m', '3m', '1m']
    r_symbols = requests.get('https://api.iextrading.com/1.0/ref-data/symbols')
    json_symbols = r_symbols.json()
    symbols = [item['symbol'] for item in json_symbols if item['isEnabled']]
    if timeframe in valid_times and symbol in symbols:
        r_summ = requests.get('https://api.iextrading.com/1.0/stock/{}/company'.format(symbol.lower()))
        json_summ = r_summ.json()
        r_time = requests.get('https://api.iextrading.com/1.0/stock/{}/chart/{}'.format(symbol.lower(), timeframe))
        json_time = r_time.json()
        if 'y' in timeframe:
            if timeframe != 'ytd':
                # 251 short work year
                coherence = (len(json_time)/251)/(int(re.findall('\d*', timeframe)[0]))
                difference = (len(json_time)/251)-(int(re.findall('\d*', timeframe)[0]))
            else:
                coherence = (len(json_time)/251)
                difference = (len(json_time)/251)-1
        else:
            coherence = (len(json_time)/22)/(int(re.findall('\d*', timeframe)[0]))
            difference = (len(json_time)/22)-(int(re.findall('\d*', timeframe)[0]))
        print('Coherence: {}, Difference: {}'.format(round(coherence, 5), round(difference, 5)))
        if coherence < 0.95:
            print('Warning: Appears some is missing...')
        # Summary Vars Wanted
        stock_document = {i:json_summ[i] for i in summ_wanted}
        stock_document['timeframe'] = timeframe
        print('retrieving {} data for {}.\n'.format(stock_document['companyName'], timeframe))
        # Time Vars Wanted
        time_wanted = ['date', 'open', 'high', 'low', 'close', 'volume', 'change', 'vwap']
        # Nested Document Structure under 'data'
        documents = []
        nan_count = 0
        for item in json_time:
            document = {}
            if all([True for i in time_wanted if i in item.keys()]):
                # Instead of the traditional close-close, open-open differencing, 
                # lets make a daily "profile" to streamline data gathering
                document['date'] = datetime.fromisoformat(item['date'])
                document['open_close'] = item['open'] - item['close']
                document['high_low'] = item['high'] - item['low']
                # This will be correlated with vwap slightly...
                document['volume'] = item['volume']
                document['close'] = item['close']
                # Assuming close to close from previous day
                document['change'] = item['change']
                # Since we trade at night, lets give the bot some info on "trend"
                document['close_vwap'] = item['close'] - item['vwap']
            else:
                document['date'] = datetime.fromisoformat(item['date'])
                inserts = ['date', 'open_close', 'high_low', 'volume', 'close', 'change', 'close-vwap']
                document.update({i:0 for i in inserts})
                nan_count += 1
                print('NaN Item... \n', item)
            documents.append(document)
        stock_document['data'] = documents
        return stock_document
    else:
        print('Something wrong...')
    
if __name__ == "__main__":
    '''NOTE: Running mongo.py requires database made by running iex.py '''
    print('This will replicate my DB...\n')
    choice = input('Want to continue? ').lower()
    if choice == 'y':
        import pymongo
        from tqdm import tqdm
        from time import sleep
        symbols = ['AAPL', 'MSFT', 'AMZN', 'INTC', 'AMD']
        print('creating: {}\n'.format(symbols))
        list_documents = []
        for i in tqdm(symbols):
            list_documents.append(make_doc_from_API(i, '5y'))
            # Always use protection...from DoS
            sleep(10)
        # Code to add docs to DB
        client = pymongo.MongoClient()
        equities = client.equities
        stocks = equities.stocks
        print('connected to {}\n'.format(stocks))
        if list_documents:
            print('inserting: {}'.format(symbols))
            for i in tqdm(list_documents):
                stocks.insert_one(i)
    else:
        print('abandoning...')

    '''NOTE: Coherence and should be close to 1 [i.e. 1.00239], its a generalized workmonth/workyear comparison.
    Difference should be close to zero [i.e. 0.01195].
    If you want to see if this worked without diving into mongo queries...
    Go to terminal, type:
    >mongo
    >show dbs (should see equities)
    >use equities
    >show collections (confirms collection exists)
    >db.stocks.find({symbol:{$exists:1}}, {symbol:1})
    '''