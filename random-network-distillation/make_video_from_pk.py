import os
import pickle
import imageio
import numpy as np

from atari_wrappers import make_atari

PK_PATH = "/workspace/runs/rnd_video_test/videos_0.pk"
OUT_PATH = "/workspace/runs/rnd_video_test/video_1_agent_non_entraine.mp4"

ENV_ID = "MontezumaRevengeNoFrameskip-v4"
FPS = 30

with open(PK_PATH, "rb") as f:
    episode = pickle.load(f)

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

for i, action in enumerate(episode["acs"]):
    obs, reward, done, info = env.step(int(action))
    frame = unwrapped_env.render(mode="rgb_array")
    frames.append(frame)
    ret += reward

print("Frames:", len(frames))
print("Return rejoué:", ret)
print("Return enregistré:", episode["info"]["r"])
print("Salles visitées:", episode["info"]["visited_rooms"])

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
imageio.mimwrite(OUT_PATH, frames, fps=FPS)

print("Vidéo créée :", OUT_PATH)
