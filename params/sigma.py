import numpy as np
import pandas as pd

from config import GRID_FREQUENCY, BINANCE_PATH, SIGMA_WINDOW, HALF_LIFE, EWMA_WARMUP, TRAIN_END, to_seconds

def build_mid(bt, grid_frequency = GRID_FREQUENCY) -> tuple[pd.DataFrame, pd.DataFrame]:
    mid_path = BINANCE_PATH / f'mid_{grid_frequency}.parquet'
    deduped_path = BINANCE_PATH / 'deduped.parquet'

    if mid_path.exists() and deduped_path.exists():
        mid = pd.read_parquet(mid_path)
        deduped = pd.read_parquet(deduped_path)
    else:
        bt_c = bt[['update_id', 'best_bid_price', 'best_ask_price', 'transaction_time']].copy()
        bt_c = bt_c.sort_values('update_id')
        bt_c['mid_price'] = (bt_c['best_bid_price'] + bt_c['best_ask_price']) / 2
        bt_c = bt_c.set_index(pd.to_datetime(bt_c['transaction_time'], unit = 'ms'))

        deduped_bools = bt_c.index.duplicated(keep = 'last') # keep only last trade for every ms
        deduped = bt_c[~deduped_bools]
        deduped[['mid_price']].to_parquet(deduped_path)

        mid = deduped.resample(grid_frequency).ffill() # refactor to grid frequency
        mid[['mid_price']].to_parquet(mid_path)

    return mid[['mid_price']], deduped[['mid_price']]

def estimate_rolling_vol(mid, grid_frequency = GRID_FREQUENCY, sigma_window = SIGMA_WINDOW) -> pd.Series:
    mid = mid['mid_price']
    dS = mid.diff() # increments

    dt = to_seconds(grid_frequency)
    var = dS.rolling(sigma_window).var() / dt
    sigma = np.sqrt(var) 
    cutoff = sigma.index[0] + pd.Timedelta(sigma_window)
    sigma.loc[:cutoff] = np.nan

    return sigma

def estimate_ewma_vol(mid, grid_frequency = GRID_FREQUENCY, half_life = HALF_LIFE, warmup = EWMA_WARMUP) -> pd.Series:
    mid = mid['mid_price']
    dS = mid.diff()

    dt = to_seconds(grid_frequency)
    var = dS.ewm(halflife = half_life / dt).var() / dt # Var(delta(S)) / delta(T) yields $^2 / second
    sigma = np.sqrt(var) # $ / sqrt(second)
    cutoff = sigma.index[0] + warmup * pd.Timedelta(half_life)
    sigma.loc[:cutoff] = np.nan

    return sigma

def measure_sigma(bt, grid_frequency, train_end = TRAIN_END) -> float:
    # sigma: standard deviation of difference in mid-prices

    mid, _ = build_mid(bt, grid_frequency = grid_frequency)

    mid = mid['mid_price']
    mid = slice_window(mid, end = train_end)
    dS = mid.diff()
    dt = to_seconds(grid_frequency)
    var = dS.var() / dt
    sigma = np.sqrt(var) 

    return sigma

def return_grid_sigmas(bt) -> dict:
    frequency_map = {}
    grid_frequencies = ['100ms', '500ms', '1s', '5s', '30s', '1min', '2min', '5min']
    for grid_frequency in grid_frequencies:
        sigma = measure_sigma(bt, grid_frequency)
        frequency_map[grid_frequency] = sigma
    
    return frequency_map

def slice_window(param, start = None, end = TRAIN_END) -> pd.Series:
    return param.loc[start:end]