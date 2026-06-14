import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio

EP_DIR = Path("/workspace/results_rnd_classic_15M/interesting_episodes")
OUT = Path("/workspace/results_rnd_classic_15M/RND_classic_interesting_episodes_metrics.mp4")
TMP = Path("/workspace/results_rnd_classic_15M/tmp_video_frames")
TMP.mkdir(exist_ok=True)

episodes = sorted(EP_DIR.glob("*.pk"))

frames = []
fps = 8

for epi_idx, path in enumerate(episodes):
    with open(path, "rb") as f:
        ep = pickle.load(f)

    rews_ext = np.array(ep["rews_ext"])
    rews_int = np.array(ep["rews_int_norm"])
    actions = np.array(ep["acs"])

    cum_ext = np.cumsum(rews_ext)
    cum_int = np.cumsum(rews_int)

    total_ext = cum_ext[-1] if len(cum_ext) else 0
    total_int = cum_int[-1] if len(cum_int) else 0

    T = len(rews_ext)
    step_points = np.linspace(1, T, min(120, T)).astype(int)

    for frame_id, t in enumerate(step_points):
        fig = plt.figure(figsize=(12, 7))

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(cum_ext[:t], label="Cumulative Extrinsic Reward")
        ax1.set_title(f"RND classique — Episode important {epi_idx+1}/{len(episodes)}")
        ax1.set_ylabel("Extrinsic")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(cum_int[:t], label="Cumulative Intrinsic Reward")
        ax2.set_ylabel("Intrinsic")
        ax2.grid(True)
        ax2.legend(loc="upper left")

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(actions[:t], label="Actions")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Action ID")
        ax3.grid(True)
        ax3.legend(loc="upper left")

        fig.text(
            0.02, 0.02,
            f"File: {path.name} | Step: {t}/{T} | Final ext: {total_ext:.1f} | Final int: {total_int:.2f}",
            fontsize=10
        )

        img_path = TMP / f"frame_{len(frames):06d}.png"
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(img_path, dpi=120)
        plt.close(fig)
        frames.append(img_path)

with imageio.get_writer(OUT, fps=fps) as writer:
    for img_path in frames:
        writer.append_data(imageio.imread(img_path))

print("Vidéo créée :", OUT)
print("Nombre de frames :", len(frames))
