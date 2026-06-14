import os
import pickle
from collections import defaultdict

import numpy as np

from baselines import logger
from mpi4py import MPI


class Recorder(object):
    def __init__(self, nenvs, score_multiple=1):
        self.episodes = [defaultdict(list) for _ in range(nenvs)]
        self.total_episodes = 0
        self.filename = self.get_filename()
        self.score_multiple = score_multiple

        # Critères objectifs pour comparaison RND vs RND + decay
        self.best_ret = -float("inf")
        self.best_n_rooms = 0
        self.seen_rooms = set()

        self.interesting_dir = os.path.join(logger.get_dir(), "interesting_episodes")
        os.makedirs(self.interesting_dir, exist_ok=True)

    def record(self, bufs, infos):
        for env_id, ep_infos in enumerate(infos):
            left_step = 0
            done_steps = sorted(ep_infos.keys())

            for right_step in done_steps:
                for key in bufs:
                    self.episodes[env_id][key].append(
                        bufs[key][env_id, left_step:right_step].copy()
                    )

                self.record_episode(env_id, ep_infos[right_step])

                left_step = right_step

                for key in bufs:
                    self.episodes[env_id][key].clear()

            for key in bufs:
                self.episodes[env_id][key].append(
                    bufs[key][env_id, left_step:].copy()
                )

    def record_episode(self, env_id, info):
        self.total_episodes += 1

        save, reasons = self.episode_worth_saving(env_id, info)

        if save:
            episode = {}

            for key in self.episodes[env_id]:
                episode[key] = np.concatenate(self.episodes[env_id][key])

            info["env_id"] = env_id
            info["save_reasons"] = reasons
            info["total_episode_index"] = self.total_episodes
            episode["info"] = info

            # Fichier global compatible avec replayer/make_video_from_pk
            with open(self.filename, "ab") as f:
                pickle.dump(episode, f, protocol=-1)

            # Fichier séparé plus lisible pour retrouver les meilleurs épisodes
            safe_reasons = "_".join(reasons)
            out_name = "episode_{:06d}_env{}_{}.pk".format(
                self.total_episodes,
                env_id,
                safe_reasons
            )
            out_path = os.path.join(self.interesting_dir, out_name)

            with open(out_path, "wb") as f:
                pickle.dump(episode, f, protocol=-1)

            print(
                "[RECORDER] Saved episode {} | env={} | r={} | rooms={} | reasons={}".format(
                    self.total_episodes,
                    env_id,
                    info.get("r", None),
                    len(info.get("places", [])),
                    reasons,
                ),
                flush=True,
            )

    def episode_worth_saving(self, env_id, info):
        reasons = []

        ret = float(info.get("r", 0.0))

        places = info.get("places", set())
        if places is None:
            places = set()

        places = set(places)
        n_rooms = len(places)

        # Critère 1 : nouveau meilleur score extrinsèque
        if ret > self.best_ret:
            self.best_ret = ret
            reasons.append("best_ret")

        # Critère 2 : nouveau maximum de salles visitées dans un épisode
        if n_rooms > self.best_n_rooms:
            self.best_n_rooms = n_rooms
            reasons.append("best_rooms")

        # Critère 3 : découverte d'au moins une nouvelle salle globale
        new_rooms = places - self.seen_rooms
        if len(new_rooms) > 0:
            self.seen_rooms.update(new_rooms)
            reasons.append("new_room")

        return len(reasons) > 0, reasons

    def get_filename(self):
        filename = os.path.join(
            logger.get_dir(),
            "videos_{}.pk".format(MPI.COMM_WORLD.Get_rank())
        )
        return filename
