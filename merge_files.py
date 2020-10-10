import sys
import os
from os import path
import pandas as pd
import numpy as np
from pathlib import Path


output_dir = Path(
    'D:\\Projects\\Spotify-Popularity-Predictor\\filtered_dataset\\')
outputfile = 'merged_without_duplicates.csv'
output1 = 'merged_with_duplicates.csv'
filenames = os.listdir(output_dir)
path = output_dir/filenames[0]
filedf = pd.read_csv(path)

df = pd.DataFrame(filedf)

# path = 'filtered_dataset\'
for filename in filenames[1:]:
    path = output_dir/filename
    filedf = pd.read_csv(path)
    df = pd.concat([df,filedf])

output = output_dir/output1
if os.path.exists(output):
    os.remove(output)

df.to_csv(output, index=False)

print(len(df.index))
# uniq_df = df.trackid.unique()
uniq_df = df.drop_duplicates(subset=["trackid"])
# uniq_df = pd.DataFrame(uniq_df)

output = output_dir/outputfile
if os.path.exists(output):
    os.remove(output)

uniq_df.to_csv(output, index=False)

print(len(uniq_df.index))
