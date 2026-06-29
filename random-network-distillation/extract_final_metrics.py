import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def parse_log(log_path):
    metrics = {}

    if not log_path.exists():
        return metrics

    text = log_path.read_text(errors="ignore")

    patterns = {
        "best_return_log": r"Best return\s+([-\w\.]+)",
        "n_rooms_log": r"n_rooms\s+\[?([0-9]+)\]?",
        "eprew_log": r"eprews\s+\[?([-\w\.]+)\]?",
    }

    for key, pat in patterns.items():
        vals = re.findall(pat, text)
        if vals:
            v = safe_float(vals[-1])
            if v is not None:
                metrics[key] = v

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    progress_path = run_dir / "progress.csv"
    log_path = run_dir / "log.txt"

    out = {
        "method": args.method,
        "seed": args.seed,
        "run_dir": str(run_dir),
    }

    if progress_path.exists() and progress_path.stat().st_size > 0:
        try:
            df = pd.read_csv(progress_path)
            df = df.replace([np.inf, -np.inf], np.nan)

            if len(df) > 0:
                for col in df.columns:
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(series) == 0:
                        continue

                    out[f"{col}_last"] = safe_float(series.iloc[-1])
                    out[f"{col}_max"] = safe_float(series.max())
                    out[f"{col}_mean"] = safe_float(series.mean())

                if "best_ret" in df.columns:
                    out["best_return"] = safe_float(pd.to_numeric(df["best_ret"], errors="coerce").replace([np.inf, -np.inf], np.nan).max())

                if "n_rooms" in df.columns:
                    out["n_rooms"] = safe_float(pd.to_numeric(df["n_rooms"], errors="coerce").replace([np.inf, -np.inf], np.nan).max())

                if "time_elapsed" in df.columns:
                    out["time_elapsed"] = safe_float(pd.to_numeric(df["time_elapsed"], errors="coerce").max())

                if "n_updates" in df.columns:
                    out["updates"] = safe_float(pd.to_numeric(df["n_updates"], errors="coerce").max())

                for key in [
                    "eprew",
                    "retextmean",
                    "retintmean",
                    "opt_auxloss",
                    "rnd_decay",
                    "int_coeff_effective",
                    "recent_ext_mean",
                ]:
                    if key in df.columns:
                        series = pd.to_numeric(df[key], errors="coerce").dropna()
                        if len(series) > 0:
                            out[key] = safe_float(series.iloc[-1])

        except Exception as e:
            out["progress_error"] = str(e)

    out.update(parse_log(log_path))

    if "best_return" not in out and "best_return_log" in out:
        out["best_return"] = out["best_return_log"]

    if "n_rooms" not in out and "n_rooms_log" in out:
        out["n_rooms"] = out["n_rooms_log"]

    output_path = run_dir / "final_metrics.json"
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
