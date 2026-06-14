import time
import os
from datetime import datetime
import pandas as pd

PROGRESS = "/workspace/runs/rnd_classic_15M/progress.csv"
OUT = "/workspace/runs/rnd_classic_15M/metrics_every_20min.csv"

COLS = [
    "timestamp",
    "tcount",
    "time_elapsed",
    "tps",
    "best_ret",
    "eprew",
    "rewtotal",
    "n_rooms",
    "rooms",
    "eplen",
    "epcount",
    "retintmean",
    "rewintmean_norm",
    "rewintmean_unnorm",
    "retextmean",
    "opt_auxloss",
    "opt_ent",
    "opt_approxkl",
    "opt_clipfrac"
]

while True:
    try:
        if os.path.exists(PROGRESS) and os.path.getsize(PROGRESS) > 0:
            df = pd.read_csv(PROGRESS)
            if len(df) > 0:
                last = df.iloc[-1].to_dict()
                row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

                for c in COLS:
                    if c != "timestamp":
                        row[c] = last.get(c, "")

                out_df = pd.DataFrame([row])
                header = not os.path.exists(OUT)
                out_df.to_csv(OUT, mode="a", header=header, index=False)

                print("Saved:", row)

    except Exception as e:
        print("Monitor error:", e)

    time.sleep(20 * 60)
