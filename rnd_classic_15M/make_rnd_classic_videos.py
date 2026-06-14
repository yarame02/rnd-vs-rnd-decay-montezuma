import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

BASE = Path("/workspace/results_rnd_classic_15M")
EP_DIR = BASE / "interesting_episodes"
TMP = BASE / "video_frames"
TMP.mkdir(exist_ok=True)

BEST_RETURN = 1400
ROOMS_VISITED = 11
TIMESTEPS = "12.97M"

def load_episode(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def episode_score(ep):
    return float(np.sum(ep["rews_ext"]))

def make_frames_for_episode(ep, title, out, max_frames=160, fps=10):
    rews_ext = np.array(ep["rews_ext"])
    rews_int = np.array(ep["rews_int_norm"])
    actions = np.array(ep["acs"])

    cum_ext = np.cumsum(rews_ext)
    cum_int = np.cumsum(rews_int)

    T = len(rews_ext)
    points = np.linspace(1, T, min(max_frames, T)).astype(int)

    frame_paths = []

    for idx, t in enumerate(points):
        fig = plt.figure(figsize=(13, 7))

        fig.suptitle(title, fontsize=16, fontweight="bold")

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(cum_ext[:t])
        ax1.set_ylabel("Reward extrinsèque cumulée")
        ax1.grid(True)
        ax1.set_title(f"Score cumulé : {cum_ext[t-1]:.0f} / final {cum_ext[-1]:.0f}")

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(cum_int[:t])
        ax2.set_ylabel("Reward intrinsèque cumulée")
        ax2.grid(True)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(actions[:t])
        ax3.set_ylabel("Action")
        ax3.set_xlabel(f"Step {t}/{T}")
        ax3.grid(True)

        fig.text(
            0.02, 0.02,
            f"RND classique | Best Return global = {BEST_RETURN} | Rooms Visited = {ROOMS_VISITED} | Timesteps = {TIMESTEPS}",
            fontsize=10
        )

        frame = TMP / f"{out.stem}_{idx:05d}.png"
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        plt.savefig(frame, dpi=120)
        plt.close(fig)
        frame_paths.append(frame)

    with imageio.get_writer(out, fps=fps) as writer:
        for frame in frame_paths:
            writer.append_data(imageio.imread(frame))

    print("Vidéo créée :", out)

def make_highlights_video(progress_csv, out, fps=10, max_frames=180):
    df = pd.read_csv(progress_csv)

    cols_needed = ["tcount", "best_ret", "n_rooms", "eprew", "retintmean", "retextmean"]
    df = df[[c for c in cols_needed if c in df.columns]].dropna()

    if len(df) > max_frames:
        idx = np.linspace(0, len(df)-1, max_frames).astype(int)
        df = df.iloc[idx]

    frame_paths = []

    for i, (_, row) in enumerate(df.iterrows()):
        fig = plt.figure(figsize=(13, 7))
        fig.suptitle("RND classique — Highlights de l'entraînement", fontsize=16, fontweight="bold")

        current = df.iloc[:i+1]

        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(current["tcount"], current["best_ret"])
        ax1.set_title("Best Return")
        ax1.grid(True)

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(current["tcount"], current["n_rooms"])
        ax2.set_title("Nombre de salles visitées")
        ax2.grid(True)

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(current["tcount"], current["eprew"])
        ax3.set_title("Episode Reward")
        ax3.grid(True)

        ax4 = plt.subplot(2, 2, 4)
        if "retintmean" in current.columns:
            ax4.plot(current["tcount"], current["retintmean"], label="Intrinsic")
        if "retextmean" in current.columns:
            ax4.plot(current["tcount"], current["retextmean"], label="Extrinsic")
        ax4.set_title("Intrinsic vs Extrinsic return")
        ax4.grid(True)
        ax4.legend()

        fig.text(
            0.02, 0.02,
            f"Final: Best Return = {BEST_RETURN} | Rooms Visited = {ROOMS_VISITED} | Timesteps = {TIMESTEPS}",
            fontsize=11
        )

        frame = TMP / f"{out.stem}_{i:05d}.png"
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        plt.savefig(frame, dpi=120)
        plt.close(fig)
        frame_paths.append(frame)

    with imageio.get_writer(out, fps=fps) as writer:
        for frame in frame_paths:
            writer.append_data(imageio.imread(frame))

    print("Vidéo créée :", out)

episodes = sorted(EP_DIR.glob("*.pk"))
if not episodes:
    raise RuntimeError("Aucun épisode trouvé dans interesting_episodes")

# Vidéo 1 : meilleur épisode selon reward extrinsèque
scored = []
for p in episodes:
    ep = load_episode(p)
    scored.append((episode_score(ep), p))

best_score, best_path = max(scored, key=lambda x: x[0])
best_ep = load_episode(best_path)

make_frames_for_episode(
    best_ep,
    f"RND_CLASSIC_BEST_EPISODE — {best_path.name} — score épisode {best_score:.0f}",
    BASE / "RND_CLASSIC_BEST_EPISODE.mp4",
    max_frames=180,
    fps=10
)

# Vidéo 2 : découvertes importantes, concaténées
discovery_files = [p for p in episodes if "new_room" in p.name or "best_rooms" in p.name]
if not discovery_files:
    discovery_files = episodes

all_frame_paths = []
out = BASE / "RND_CLASSIC_DISCOVERIES.mp4"

for j, p in enumerate(discovery_files):
    ep = load_episode(p)
    rews_ext = np.array(ep["rews_ext"])
    rews_int = np.array(ep["rews_int_norm"])
    actions = np.array(ep["acs"])
    cum_ext = np.cumsum(rews_ext)
    cum_int = np.cumsum(rews_int)
    T = len(rews_ext)
    points = np.linspace(1, T, min(60, T)).astype(int)

    for idx, t in enumerate(points):
        fig = plt.figure(figsize=(13, 7))
        fig.suptitle(f"RND classique — Découverte importante {j+1}/{len(discovery_files)}", fontsize=16, fontweight="bold")

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(cum_ext[:t])
        ax1.set_title(f"{p.name} | Score épisode final : {cum_ext[-1]:.0f}")
        ax1.set_ylabel("Extrinsic")
        ax1.grid(True)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(cum_int[:t])
        ax2.set_ylabel("Intrinsic")
        ax2.grid(True)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(actions[:t])
        ax3.set_ylabel("Action")
        ax3.set_xlabel(f"Step {t}/{T}")
        ax3.grid(True)

        fig.text(
            0.02, 0.02,
            f"Rooms Visited final = {ROOMS_VISITED} | Rooms = 0,1,2,3,4,5,9,10,11,12,19",
            fontsize=10
        )

        frame = TMP / f"{out.stem}_{len(all_frame_paths):05d}.png"
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        plt.savefig(frame, dpi=120)
        plt.close(fig)
        all_frame_paths.append(frame)

with imageio.get_writer(out, fps=10) as writer:
    for frame in all_frame_paths:
        writer.append_data(imageio.imread(frame))

print("Vidéo créée :", out)

# Vidéo 3 : highlights globaux
make_highlights_video(
    BASE / "progress.csv",
    BASE / "RND_CLASSIC_HIGHLIGHTS.mp4",
    fps=10,
    max_frames=180
)

print("Terminé.")
