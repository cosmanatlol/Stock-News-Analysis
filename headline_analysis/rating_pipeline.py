import pandas as pd
from Filter import filtering
from llm_models import headline_rating, relevance_extractor
import time

def rating_pipeline(headlines_path, stocks, store_path, start = None, end = None, merge_path = None):
    """
    headlines_path: str path to csv file with headlines
    stocks: dictionary with keys as sectors and values as list of stocks
    store_path: str path to store the final dataframe
    merge_path: str path to csv file to merge with the final dataframe
    start: timestamp for start date for the headlines, default None which uses the minimum date
    end: timestamp for end date for the headlines, default None which uses the maximum date

    output: dataframe with stock, date, and rating. Rating is for the prediction fir the day after the date
    """
    data = pd.read_csv(headlines_path)
    data['date'] = pd.to_datetime(data['date']).dt.floor('D')
    if not start:
        start = data['date'].min()
    if not end:
        end = data['date'].max()
    data = data[(data['date'] >= start) & (data['date'] <= end)]
    date_range = data['date'].unique()
    final_data = []
    for days in date_range:
        data_splice = data[data['date'] == days]
        data_splice = filtering(data_splice)
        for sector in list(stocks.keys()):
            relevance_list = []
            for x in data_splice['headline']:
                relevance_list += [relevance_extractor(x, sector)]
                time.sleep(1)
            headlines = data_splice[relevance_list]['headline'].tolist()
            for stock in stocks[sector]:
                rating = headline_rating(headlines, stock)
                final_data.append([stock, days, rating])
                time.sleep(1)
    if merge_path:
        merge_data = pd.read_csv(merge_path)
        final_df = pd.DataFrame(final_data, columns=['stock', 'date', 'rating'])
        final_df = pd.concat([merge_data, final_df])
        final_df.to_csv(store_path)
        return final_df
    final_df = pd.DataFrame(final_data, columns=['stock', 'date', 'rating'])
    final_df.to_csv(store_path)
    return final_df


    

