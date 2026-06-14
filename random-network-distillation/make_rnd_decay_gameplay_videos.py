import os
import pickle
import imageio
import numpy as np
from pathlib import Path

from atari_wrappers import make_atari

PK_PATH = "/workspace/runs/rnd_decay_15M/videos_0.pk"
OUT_DIR = Path("/workspace/runs/rnd_decay_15M/gameplay_videos")
OUT_DIR.mkdir(exist_ok=True)

ENV_ID = "MontezumaRevengeNoFrameskip-v4"
FPS = 30
MAX_EPISODES = 20

def load_episodes(pk_path):
    episodes = []
    with open(pk_path, "rb") as f:
        while True:
            try:
                episodes.append(pickle.load(f))
            except EOFError:
                break
    return episodes

def replay_episode_to_mp4(episode, out_path, fps=30):
    env = make_atari(ENV_ID, max_episode_steps=4500)
    unwrapped_env = env.unwrapped

    info0 = episode["info"]

    if "rng_at_episode_start" in info0:
        random_state = info0["rng_at_episode_start"]
        unwrapped_env.np_random.set_state(random_state.get_state())
        if hasattr(unwrapped_env, "scene"):
            unwrapped_env.scene.np_random.set_state(random_state.get_state())

    env.reset()

    frames = []
    ret = 0

    for action in episode["acs"]:
        obs, reward, done, info = env.step(int(action))
        frame = unwrapped_env.render(mode="rgb_array")
        frames.append(frame)
        ret += reward
        if done:
            break

    env.close()

    imageio.mimwrite(str(out_path), frames, fps=fps)

    return ret, len(frames), info0

episodes = load_episodes(PK_PATH)

print("Nombre episodes sauvegardes:", len(episodes))

scored = []
for i, ep in enumerate(episodes):
    info = ep["info"]
    score = float(info.get("r", 0))
    n_rooms = len(info.get("places", []))
    reasons = "_".join(info.get("save_reasons", ["episode"]))
    scored.append((score, n_rooms, i, reasons))

scored_sorted = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)

# 1. Meilleur épisode
best_score, best_rooms, best_i, reasons = scored_sorted[0]
out = OUT_DIR / "RND_DECAY_BEST_EPISODE_GAMEPLAY.mp4"
ret, nframes, info = replay_episode_to_mp4(episodes[best_i], out, FPS)
print("BEST EPISODE:", out, "ret=", ret, "frames=", nframes, "info_r=", info.get("r"), "rooms=", info.get("places"))

# 2. Episodes de découverte / new rooms
selected = []
seen = set()

for score, n_rooms, i, reasons in scored_sorted:
    info = episodes[i]["info"]
    places = set(info.get("places", []))
    if len(places - seen) > 0 or "new_room" in reasons or "best_rooms" in reasons:
        selected.append(i)
        seen.update(places)
    if len(selected) >= 8:
        break

disc_frames = []

for j, i in enumerate(selected):
    tmp = OUT_DIR / f"tmp_discovery_{j:02d}.mp4"
    ret, nframes, info = replay_episode_to_mp4(episodes[i], tmp, FPS)
    print("DISCOVERY:", tmp, "ret=", ret, "rooms=", info.get("places"))

    reader = imageio.get_reader(str(tmp))
    for frame in reader:
        disc_frames.append(frame)
    reader.close()

out = OUT_DIR / "RND_DECAY_DISCOVERIES_GAMEPLAY.mp4"
imageio.mimwrite(str(out), disc_frames, fps=FPS)
print("DISCOVERIES VIDEO:", out, "frames=", len(disc_frames))

# 3. Highlights courts : meilleur épisode + quelques découvertes
highlight_indices = [best_i] + selected[:4]
highlight_frames = []

for j, i in enumerate(highlight_indices):
    tmp = OUT_DIR / f"tmp_highlight_{j:02d}.mp4"
    ret, nframes, info = replay_episode_to_mp4(episodes[i], tmp, FPS)

    reader = imageio.get_reader(str(tmp))
    frames = list(reader)
    reader.close()

    # garder maximum 20 secondes par épisode
    max_keep = min(len(frames), FPS * 20)
    highlight_frames.extend(frames[:max_keep])

out = OUT_DIR / "RND_DECAY_HIGHLIGHTS_GAMEPLAY.mp4"
imageio.mimwrite(str(out), highlight_frames, fps=FPS)
print("HIGHLIGHTS VIDEO:", out, "frames=", len(highlight_frames))

print("Terminé.")
