# append all the csv files in the raw/pitcher directory
# into a single DataFrame

import os
import pandas as pd

def combine_pitcher_data():
    """
    Combine all pitcher Statcast data into a single DataFrame.
    
    Returns:
    DataFrame: The combined data
    """
    # Get all files in the directory
    files = [f for f in os.listdir("../data/raw/pitcher") if f.endswith(".csv")]
    
    # Read all files into a list
    data = [pd.read_csv(f"../data/raw/pitcher/{f}") for f in files]
    
    # Concatenate the list into a single DataFrame
    return pd.concat(data, ignore_index=True)