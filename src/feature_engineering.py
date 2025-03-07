import os
import pandas as pd



def apply_all_features(df):
    """Apply all feature engineering in a single efficient pass"""
    # Make just one copy at the start
    df = df.copy()
    
    # Pre-process date column once
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # At-bat ID
    df['at_bat_id'] = df['game_pk'].astype(str) + df['at_bat_number'].astype(str)
    
    # Month of season
    df['month'] = df['game_date'].dt.month
    
    # Calculate score difference once (used by multiple features)
    df['score_diff'] = df['fld_score'] - df['bat_score']
    
    # RISP calculation
    df['risp'] = (df['on_2b'] > 0) | (df['on_3b'] > 0)
    
    # Late-inning clutch situations
    late_close = (df['inning'] >= 7) & (df['score_diff'].between(0, 3))
    late_risp = (df['inning'] >= 7) & (df['score_diff'].between(0, 5)) & df['risp']
    df['clutch'] = (late_close | late_risp).astype(int)
    
    # Blowout situations
    df['blowout'] = (df['score_diff'].abs() >= 10).astype(int)
    
    # All-star break
    all_star = pd.to_datetime('2021-07-14')
    df['post_allstar_break'] = (df['game_date'] >= all_star).astype(int)
    
    return df