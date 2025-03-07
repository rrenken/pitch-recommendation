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


def drop_null_columns(data):
    """
    Drop unnecessary columns from the data.
    
    Args:
    data (DataFrame): The combined data
    
    Returns:
    DataFrame: The data with columns dropped
    """
    # Drop columns that are not useful for the model
    return data.drop(columns=['spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 
                              'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated',
                               'umpire', 'sv_id',
                               'description','des'])




def handle_missing(df):
    """Handle null values in a baseball-aware manner"""
    df = df.copy()
    
    # Fill categorical nulls with special values
    cat_fill_na = ['pitch_type', 'events', 'description', 'zone', 
                   'des', 'hit_location', 'bb_type', 'launch_speed_angle', 
                   'pitch_name', 
                    'if_fielding_alignment', 'of_fielding_alignment' 
                    
                    ]

    for col in cat_fill_na:
        if col in df.columns:
            df[col] = df[col].fillna('N/a')
            
    # Zero-fill numerical features where zero means "didn't happen"
    cont_fill_na = ['release_speed', 'release_pos_x', 'release_pos_z', 
                    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'hc_x', 'hc_y',
                    'vx0', 'vy0', 
                    'vz0', 'ax', 'ay', 
                    'az', 'sz_top', 'sz_bot', 'hit_distance_sc', 
                    'launch_speed', 'launch_angle', 'effective_speed', 
                    'release_spin_rate', 'release_extension', 
                    'release_pos_y', 'estimated_ba_using_speedangle',
                    'estimated_woba_using_speedangle', 'woba_value',
                    'woba_denom', 'babip_value', 'iso_value', 'spin_axis',
                    'delta_home_win_exp', 'delta_run_exp', 'bat_speed' ,
                    'swing_length', 'estimated_slg_using_speedangle',
                    'delta_pitcher_run_exp', 'hyper_speed',
                    'pitcher_days_since_prev_game', 
                    'batter_days_since_prev_game',
                    'pitcher_days_until_next_game',
                    'batter_days_until_next_game',
                    'api_break_z_with_gravity',
                    'api_break_x_arm',
                    'api_break_x_batter_in',
                    'arm_angle', 'home_win_exp',
                    'bat_win_exp', 'on_3b', 'on_2b',
                   'on_1b'

                    ]

    for col in cont_fill_na:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df