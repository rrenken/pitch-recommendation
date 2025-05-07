import os
import pandas as pd
from pybaseball import statcast
from datetime import datetime
import calendar

def pitcher_year_month_statcast(year, month):
    """
    Fetch Statcast data for a specific month and year.
    
    Parameters:
    year (int): Year to fetch data for (e.g., 2023)
    month (int): Month to fetch data for (1-12)
    
    Returns:
    DataFrame: The fetched data
    """
    # Create data directory if it doesn't exist
    #os.makedirs("../data/raw/", exist_ok=True)
    
    # Calculate start and end dates for the month
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day}"
    
    # Get month name for file naming
    month_name = datetime(year, month, 1).strftime('%b').lower()
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    try:
        # Get the data from Statcast
        data = statcast(start_date, end_date)
        
        if data is not None and not data.empty:
            # Create filename
            filename = f"../data/raw/pitcher/pitcher_statcast_{year}_{month_name}.csv"
            
            # Save to CSV
            data.to_csv(filename, index=False)
            print(f"Successfully saved {len(data)} pitches to {filename}")
            return data
        else:
            print("No data returned for the specified period.")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    






def pitcher_year_month_statcast_testing(year, month):
    
    # Calculate start and end dates for the month
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day}"
    
    # Get month name for file naming
    month_name = datetime(year, month, 1).strftime('%b').lower()
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    try:
        # Get the data from Statcast
        data = statcast(start_date, end_date)
        
        if data is not None and not data.empty:
            # Create filename
            filename = f"../data/raw/testing/pitcher_statcast_{year}_{month_name}.csv"
            
            # Save to CSV
            data.to_csv(filename, index=False)
            print(f"Successfully saved {len(data)} pitches to {filename}")
            return data
        else:
            print("No data returned for the specified period.")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None