import pandas as pd, glob
from pathlib import Path

bt_path = Path('binance/bt.parquet')
at_path = Path('binance/at.parquet')

def load():
    if bt_path.exists() and at_path.exists():
        bt = pd.read_parquet(bt_path)
        at = pd.read_parquet(at_path)
    else:
        bt = pd.concat(
            [pd.read_csv(f) for f in sorted(glob.glob("binance/BTCUSDT-bookTicker-2024-03-*.zip"))],
            ignore_index = True
            )
        bt.to_parquet(bt_path)
        
        at = pd.concat(
            [pd.read_csv(f) for f in sorted(glob.glob("binance/BTCUSDT-aggTrades-2024-03-*.zip"))],
            ignore_index = True
            )
        at.to_parquet(at_path)

    return bt, at