import re
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path("/workspace/runs/rnd_decay_15M")
LOG_PATH = RUN_DIR / "train.log"
OUT_DIR = RUN_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

metrics = [
    "tcount", "time_elapsed", "tps",
    "best_ret", "eprew", "rewtotal",
    "n_rooms", "eprooms", "epcount", "eplen",
    "rnd_decay", "int_coeff_effective", "recent_ext_mean",
    "retintmean", "retextmean",
    "rewintmean_norm", "rewintmean_unnorm",
    "rewintmax_norm", "rewintmax_unnorm",
    "opt_auxloss", "opt_ent", "opt_approxkl", "opt_clipfrac",
    "ev_int", "ev_ext"
]

text = LOG_PATH.read_text(errors="ignore")
blocks = re.findall(r"-{10,}\n(.*?)\n-{10,}", text, re.S)

rows = []
for block in blocks:
    row = {}
    for line in block.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            k = m.group(1).strip()
            v = m.group(2).strip()
            row[k] = v
    if row:
        rows.append(row)

df = pd.DataFrame(rows)

def to_num(x):
    try:
        return float(str(x).replace("e+", "e").replace("e-", "e-"))
    except:
        return None

for col in df.columns:
    if col != "rooms":
        df[col] = df[col].apply(to_num)

df.to_csv(RUN_DIR / "decay_progress_full.csv", index=False)

# résumé final
final = df.dropna(how="all").iloc[-1]
summary = pd.DataFrame({
    "metric": final.index,
    "value": final.values
})
summary.to_csv(RUN_DIR / "decay_final_summary.csv", index=False)

# courbes individuelles
plot_metrics = [m for m in metrics if m in df.columns]

x = df["tcount"] if "tcount" in df.columns else range(len(df))

for m in plot_metrics:
    if m not in df.columns:
        continue
    y = df[m]
    if y.dropna().empty:
        continue

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel("tcount")
    plt.ylabel(m)
    plt.title(f"RND+Decay - {m}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{m}.png", dpi=150)
    plt.close()

# courbe combinée decay
if "rnd_decay" in df.columns and "int_coeff_effective" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(x, df["rnd_decay"], label="rnd_decay")
    plt.plot(x, df["int_coeff_effective"], label="int_coeff_effective")
    plt.xlabel("tcount")
    plt.ylabel("coefficient")
    plt.title("RND+Decay - coefficient decay")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "decay_coefficients_combined.png", dpi=150)
    plt.close()

print("Courbes générées dans :", OUT_DIR)
print("CSV complet :", RUN_DIR / "decay_progress_full.csv")
print("Résumé final :", RUN_DIR / "decay_final_summary.csv")
