import sys
import os
from os import path
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path('D:\\Projects\\Spotify-Popularity-Predictor\\filtered_dataset\\')
filenames = os.listdir(output_dir)

no_of_lines = 0
for filename in filenames:
    path = output_dir/filename
    df = pd.read_csv(path)
    no_of_lines = no_of_lines + len(df.index)
    print(f"{filename} - {len(df.index)}")

print(f"Total {no_of_lines}")
