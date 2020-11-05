import sys
import os
from os import path
import pandas as pd 
import numpy as np
from pathlib import Path


def filter_by_available_markets(df, countryname):
    ids = []
    for idx, row in df.iterrows():
        if countryname not in row.available_markets:
            ids.append(idx)

    df.drop(ids, 0, inplace=True)
    return df


filenames = os.listdir('dataset/')
output_dir = Path('D:\\Projects\\Spotify-Popularity-Predictor\\filtered_with_empty_countries_dataset\\')
for filename in filenames:
    path = 'dataset/'+filename
    output_file=output_dir/filename
    countryname = filename.split('.')[0]
    df = pd.read_csv(path)
    # cols = df.columns
    # cols = [col for col in cols if 'Unnamed' in col]
    # df.drop(cols, axis=1, inplace=True)
    index_with_nan = df.index[df.isnull().any(axis=1)]
    df.drop(index_with_nan, 0, inplace=True)    
    # df = filter_by_available_markets(df,countryname.upper())
    df.insert(loc=0, column='index', value=np.arange(len(df)))
    df.to_csv(output_file, index=False)


