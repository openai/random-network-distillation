import os
import pickle
from collections import defaultdict

import numpy as np

from baselines import logger
from mpi4py import MPI

def is_square(n):
    return n == (int(np.sqrt(n))) ** 2

class Recorder(object):
    def __init__(self, nenvs, score_multiple=1):
        self.episodes = [defaultdict(list) for _ in range(nenvs)]
        self.total_episodes = 0
        self.filename = self.get_filename()
        self.score_multiple = score_multiple

        self.all_scores = {}
        self.all_places = {}

    def record(self, bufs, infos):
        for env_id, ep_infos in enumerate(infos):
            left_step = 0
            done_steps = sorted(ep_infos.keys())
            for right_step in done_steps:
                for key in bufs:
                    self.episodes[env_id][key].append(bufs[key][env_id, left_step:right_step].copy())
                self.record_episode(env_id, ep_infos[right_step])
                left_step = right_step
                for key in bufs:
                    self.episodes[env_id][key].clear()
            for key in bufs:
                self.episodes[env_id][key].append(bufs[key][env_id, left_step:].copy())


    def record_episode(self, env_id, info):
        self.total_episodes += 1
        if self.episode_worth_saving(env_id, info):
            episode = {}
            for key in self.episodes[env_id]:
                episode[key] = np.concatenate(self.episodes[env_id][key])
            info['env_id'] = env_id
            episode['info'] = info
            with open(self.filename, 'ab') as f:
                pickle.dump(episode, f, protocol=-1)

    def get_score(self, info):
        return int(info['r']/self.score_multiple) * self.score_multiple

    def episode_worth_saving(self, env_id, info):
        if self.score_multiple is None:
            return False
        r = self.get_score(info)
        if r not in self.all_scores:
            self.all_scores[r] = 0
        else:
            self.all_scores[r] += 1
        hashable_places = tuple(sorted(info['places']))
        if hashable_places not in self.all_places:
            self.all_places[hashable_places] = 0
        else:
            self.all_places[hashable_places] += 1
        if is_square(self.all_scores[r]) or is_square(self.all_places[hashable_places]):
            return True
        if 15 in info['places']:
            return True
        return False

    def get_filename(self):
        filename = os.path.join(logger.get_dir(), 'videos_{}.pk'.format(MPI.COMM_WORLD.Get_rank()))
        return filename

