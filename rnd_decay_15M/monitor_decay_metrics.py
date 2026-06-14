import re, csv, time
from datetime import datetime
from pathlib import Path

log_path = Path("/workspace/runs/rnd_decay_15M/train.log")
out_path = Path("/workspace/runs/rnd_decay_15M/decay_metrics_every_20min.csv")

metrics = [
    "tcount","time_elapsed","tps",
    "best_ret","eprew","rewtotal",
    "n_rooms","rooms","eplen","epcount",
    "retintmean","rewintmean_norm","rewintmean_unnorm",
    "retextmean","recent_ext_mean",
    "rnd_decay","int_coeff_effective",
    "opt_auxloss","opt_ent","opt_approxkl","opt_clipfrac"
]

if not out_path.exists():
    with out_path.open("w", newline="") as f:
        csv.writer(f).writerow(["timestamp"] + metrics)

def extract_last_block(text):
    blocks = re.findall(r"-{10,}\n(.*?)\n-{10,}", text, re.S)
    return blocks[-1] if blocks else ""

def parse_block(block):
    d = {}
    for line in block.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            d[key] = val
    return d

while True:
    try:
        text = log_path.read_text(errors="ignore")
        block = extract_last_block(text)
        d = parse_block(block)

        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        row += [d.get(m, "") for m in metrics]

        with out_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

        print("Saved:", row)
    except Exception as e:
        print("Monitor error:", e)

    time.sleep(1200)
