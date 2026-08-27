"""Step-by-step jugaad-data debug."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
import warnings
warnings.filterwarnings("ignore")

_JUGAAD_CACHE = os.path.join(os.getcwd(), "topo_trader", "data", "cache", "jugaad_cache")
os.makedirs(_JUGAAD_CACHE, exist_ok=True)
os.environ["J_CACHE_DIR"] = _JUGAAD_CACHE

import jugaad_data.util as _jutil
_orig = _jutil.kw_to_fname
def _safe(**kw):
    name = _orig(**kw)
    for ch in [':', '*', '?', '<', '>', '|', '"', '\\', '/']:
        name = name.replace(ch, '-')
    return name
_jutil.kw_to_fname = _safe

print("Step 1: imports done", flush=True)

from datetime import datetime
from jugaad_data.nse import stock_df

print("Step 2: stock_df imported", flush=True)

try:
    print("Step 3: calling stock_df...", flush=True)
    df = stock_df(symbol="TCS", from_date=datetime(2022,1,1), to_date=datetime(2022,3,31), series="EQ")
    print(f"Step 4: got result, type={type(df)}", flush=True)
    print(f"Step 5: len={len(df)}", flush=True)
    if len(df) > 0:
        print(f"Step 6: cols={list(df.columns)}", flush=True)
        print(df.iloc[0].to_dict(), flush=True)
    else:
        print("Step 6: EMPTY DataFrame returned", flush=True)
    print("SUCCESS", flush=True)
except SystemExit as e:
    print(f"SYSTEMEXIT: {e}", flush=True)
except KeyboardInterrupt:
    print("KEYBOARD INTERRUPT", flush=True)
except Exception as e:
    import traceback
    print(f"EXCEPTION: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
