import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/workspace/RND_FINAL_REPORT_EXPORT/rnd_multiseed")
OUT = Path("/workspace/RND_FINAL_REPORT_EXPORT/multiseed_analysis_3methods")
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "ppo_baseline": BASE / "ppo_baseline",
    "rnd_classic": BASE / "rnd_classic",
    "rnd_decay": BASE / "rnd_decay",
}

METRICS = [
    "best_return",
    "n_rooms",
    "eprew",
    "retextmean",
    "retintmean",
    "opt_auxloss",
    "time_elapsed",
    "updates",
    "rnd_decay",
    "int_coeff_effective",
    "recent_ext_mean",
]

T_STUDENT_95_N5 = 2.776


def safe_num(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return np.nan


def load_final_metrics():
    rows = []

    for method, method_dir in METHODS.items():
        for seed in range(5):
            run_dir = method_dir / f"seed_{seed}"
            f = run_dir / "final_metrics.json"

            if not f.exists():
                print(f"[WARN] Missing {f}")
                continue

            data = json.loads(f.read_text())
            row = {
                "method": method,
                "seed": seed,
                "run_dir": str(run_dir),
            }

            for key, value in data.items():
                if key in ["method", "seed", "run_dir"]:
                    continue
                row[key] = safe_num(value)

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_seed_results_3methods.csv", index=False)
    return df


def summarize(df):
    rows = []

    for method in sorted(df["method"].dropna().unique()):
        sub = df[df["method"] == method]

        for metric in METRICS:
            if metric not in sub.columns:
                continue

            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            n = len(values)

            if n == 0:
                continue

            mean = values.mean()
            std = values.std(ddof=1) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            ci_half = T_STUDENT_95_N5 * sem if n == 5 else 1.96 * sem
            cv = (std / mean * 100) if mean != 0 else np.nan

            rows.append({
                "method": method,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "min": values.min(),
                "max": values.max(),
                "ci95_low": mean - ci_half,
                "ci95_high": mean + ci_half,
                "ci95_half_width": ci_half,
                "cv_percent": cv,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary_statistics_3methods.csv", index=False)

    latex = summary.copy()
    for col in [
        "mean", "std", "min", "max",
        "ci95_low", "ci95_high",
        "ci95_half_width", "cv_percent"
    ]:
        if col in latex.columns:
            latex[col] = latex[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )

    (OUT / "summary_statistics_3methods_latex.txt").write_text(
        latex.to_latex(index=False, escape=False)
    )

    return summary


def load_progress(method, seed):
    path = METHODS[method] / f"seed_{seed}" / "progress.csv"

    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(path)
        df["seed"] = seed
        df["method"] = method

        if "n_updates" not in df.columns:
            df["n_updates"] = np.arange(len(df))

        return df
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None


def plot_metric(metric):
    plt.figure(figsize=(8, 5))

    plotted = False

    for method in METHODS:
        seed_frames = []

        for seed in range(5):
            df = load_progress(method, seed)

            if df is None or metric not in df.columns:
                continue

            temp = df[["n_updates", metric]].copy()
            temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
            temp = temp.dropna()

            if len(temp) == 0:
                continue

            temp = temp.groupby("n_updates", as_index=False)[metric].mean()
            temp = temp.rename(columns={metric: f"seed_{seed}"})
            seed_frames.append(temp)

        if not seed_frames:
            continue

        merged = seed_frames[0]
        for sf in seed_frames[1:]:
            merged = pd.merge(merged, sf, on="n_updates", how="outer")

        merged = merged.sort_values("n_updates")
        seed_cols = [c for c in merged.columns if c.startswith("seed_")]

        y_mean = merged[seed_cols].mean(axis=1)
        y_std = merged[seed_cols].std(axis=1)

        x = merged["n_updates"]

        plt.plot(x, y_mean, label=method)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

        merged_out = merged.copy()
        merged_out["mean"] = y_mean
        merged_out["std"] = y_std
        merged_out.to_csv(
            OUT / f"curve_data_{method}_{metric}.csv",
            index=False
        )

        plotted = True

    if plotted:
        plt.xlabel("n_updates")
        plt.ylabel(metric)
        plt.title(f"{metric} mean ± std across 5 seeds")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"curve_{metric}_3methods.png", dpi=200)

    plt.close()


def main():
    df = load_final_metrics()
    print("Saved:", OUT / "per_seed_results_3methods.csv")

    summarize(df)
    print("Saved:", OUT / "summary_statistics_3methods.csv")
    print("Saved:", OUT / "summary_statistics_3methods_latex.txt")

    curve_metrics = [
        "best_ret",
        "eprew",
        "n_rooms",
        "retextmean",
        "retintmean",
        "opt_auxloss",
        "rnd_decay",
        "int_coeff_effective",
        "recent_ext_mean",
    ]

    for metric in curve_metrics:
        plot_metric(metric)

    print("Analysis complete.")
    print("Output folder:", OUT)


if __name__ == "__main__":
    main()
