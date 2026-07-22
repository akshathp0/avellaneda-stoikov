import pandas as pd, glob

from config import BINANCE_PATH

bt_path = BINANCE_PATH / 'bt.parquet'
at_path = BINANCE_PATH / 'at.parquet'

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if bt_path.exists() and at_path.exists():
        bt = pd.read_parquet(bt_path)
        at = pd.read_parquet(at_path)
    else:
        bt = pd.concat(
            [pd.read_csv(f) for f in sorted(BINANCE_PATH.glob("binance/BTCUSDT-bookTicker-2024-03-*.zip"))],
            ignore_index = True
            )
        bt.to_parquet(bt_path)
        
        at = pd.concat(
            [pd.read_csv(f) for f in sorted(BINANCE_PATH.glob("binance/BTCUSDT-aggTrades-2024-03-*.zip"))],
            ignore_index = True
            )
        at.to_parquet(at_path)

    return bt, at