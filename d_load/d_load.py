import pandas as pd, glob

def d_load():
    bt = pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob("data/BTCUSDT-bookTicker-2024-03-*.zip"))],
        ignore_index = True
        )
    
    at = pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob("data/BTCUSDT-aggTrades-2024-03-*.zip"))],
        ignore_index = True
        )
    
    return bt, at