import numpy as np
import pandas as pd
from pathlib import Path
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

SIGMA_WINDOW = config['sigma_window']
GRID_FREQUENCY = config['grid_frequency']
HALF_LIFE = config['half_life']
EWMA_WARMUP = config['ewma_warmup']
TRAINING_SLICE = config['training_slice']

mid_path = Path(f'binance/mid_{GRID_FREQUENCY}.parquet')

def to_seconds(s):
    f = pd.Timedelta(s).total_seconds()

    return f

def build_mid(bt):
    if mid_path.exists():
        mid = pd.read_parquet(mid_path)
    else:
        bt_c = bt[['update_id', 'best_bid_price', 'best_ask_price', 'transaction_time']].copy()
        bt_c = bt_c.sort_values('update_id')
        bt_c['mid_price'] = (bt_c['best_bid_price'] + bt_c['best_ask_price']) / 2
        bt_c = bt_c.set_index(pd.to_datetime(bt_c['transaction_time'], unit = 'ms'))

        deduped_bools = bt_c.index.duplicated(keep = 'last')
        deduped = bt_c[~deduped_bools]
        mid = deduped.resample(GRID_FREQUENCY).ffill() # refactor to grid frequency
        mid[['mid_price']].to_parquet(mid_path)

    return mid[['mid_price']]

def estimate_rolling_vol(mid):
    dS = mid.diff() # increments

    dt_seconds = to_seconds(GRID_FREQUENCY)
    var = dS.rolling(SIGMA_WINDOW).var() / dt_seconds # Var(delta(S)) / delta(T) yields $^2 / second
    sigma = np.sqrt(var) # $ / sqrt(second)
    cutoff = sigma.index[0] + pd.Timedelta(SIGMA_WINDOW)
    sigma.loc[:cutoff] = np.nan

    return sigma

def estimate_ewma_vol(mid):
    dS = mid.diff()

    dt_seconds = to_seconds(GRID_FREQUENCY)
    var = dS.ewm(halflife = HALF_LIFE / dt_seconds).var() / dt_seconds
    sigma = np.sqrt(var)
    cutoff = sigma.index[0] + EWMA_WARMUP * pd.Timedelta(HALF_LIFE)
    sigma.loc[:cutoff] = np.nan

    return sigma

def return_training_slice(param):
    return param.loc[:TRAINING_SLICE]