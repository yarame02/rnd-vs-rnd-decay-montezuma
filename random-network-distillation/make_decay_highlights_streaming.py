import os
import pickle
import imageio.v2 as imageio
from pathlib import Path

from atari_wrappers import make_atari

PK_PATH = "/workspace/runs/rnd_decay_15M/videos_0.pk"
OUT_DIR = Path("/workspace/runs/rnd_decay_15M/gameplay_videos")
OUT_DIR.mkdir(exist_ok=True)

ENV_ID = "MontezumaRevengeNoFrameskip-v4"
FPS = 30

def load_episodes(pk_path):
    episodes = []
    with open(pk_path, "rb") as f:
        while True:
            try:
                episodes.append(pickle.load(f))
            except EOFError:
                break
    return episodes

def replay_episode_frames(episode, max_frames=600):
    env = make_atari(ENV_ID, max_episode_steps=4500)
    unwrapped_env = env.unwrapped

    info0 = episode["info"]

    if "rng_at_episode_start" in info0:
        random_state = info0["rng_at_episode_start"]
        unwrapped_env.np_random.set_state(random_state.get_state())
        if hasattr(unwrapped_env, "scene"):
            unwrapped_env.scene.np_random.set_state(random_state.get_state())

    env.reset()

    produced = 0
    ret = 0

    for action in episode["acs"]:
        obs, reward, done, info = env.step(int(action))
        frame = unwrapped_env.render(mode="rgb_array")
        ret += reward
        yield frame
        produced += 1

        if produced >= max_frames:
            break
        if done:
            break

    env.close()

episodes = load_episodes(PK_PATH)

scored = []
for i, ep in enumerate(episodes):
    info = ep["info"]
    score = float(info.get("r", 0))
    n_rooms = len(info.get("places", []))
    reasons = "_".join(info.get("save_reasons", ["episode"]))
    scored.append((score, n_rooms, i, reasons))

scored_sorted = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)

best_i = scored_sorted[0][2]

selected = [best_i]
seen = set()

for score, n_rooms, i, reasons in scored_sorted:
    info = episodes[i]["info"]
    places = set(info.get("places", []))
    if i != best_i and (len(places - seen) > 0 or "new_room" in reasons or "best_rooms" in reasons):
        selected.append(i)
        seen.update(places)
    if len(selected) >= 5:
        break

out = OUT_DIR / "RND_DECAY_HIGHLIGHTS_GAMEPLAY.mp4"

with imageio.get_writer(str(out), fps=FPS) as writer:
    for j, i in enumerate(selected):
        ep = episodes[i]
        info = ep["info"]
        print("Ajout épisode", i, "r=", info.get("r"), "rooms=", info.get("places"))

        for frame in replay_episode_frames(ep, max_frames=600):
            writer.append_data(frame)

print("Vidéo créée :", out)
