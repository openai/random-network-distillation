import argparse
import datetime
import glob
import os
import pickle
import sys

import exptag
import ipdb
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind
from run_atari import add_env_params

seen_scores = set()


class EpisodeIterator(object):
    def __init__(self, filenames):
        if args['filter'] == 'none':
            cond = lambda info: True
        elif args['filter'] == 'rew':
            cond = lambda info: info['r'] < args['rew_max'] and (info['r'] > args['rew_min'])
        elif args['filter'] == 'room':
            def cond(info):
                return any(int(room) in args['room_number'] for room in info['places'])

        self.filenames = filenames
        self.condition = cond
        self.episode_number = 0

    def iterate(self):
        for filename in self.filenames:
            print("Opening file", filename)
            with open(filename, 'rb') as f:
                yield from self.iterate_over_episodes_in_file(f, condition=self.condition)
        raise StopIteration

    def iterate_over_episodes_in_file(self, file, condition):
        while True:
            try:
                episode = pickle.load(file)
            except:
                raise StopIteration

            info = episode['info']
            if condition(info):
                print(self.episode_number)
                self.episode_number += 1
                if self.episode_number >= args['skip']:
                    if 'obs' in episode:
                        # import ipdb; ipdb.set_trace()
                        yield episode
                    else:
                        unwrapped_env = env.unwrapped
                        if 'rng_at_episode_start' in info:
                            random_state = info['rng_at_episode_start']
                            unwrapped_env.np_random.set_state(random_state.get_state())
                            if hasattr(unwrapped_env, "scene"):
                                unwrapped_env.scene.np_random.set_state(random_state.get_state())
                        ob = env.reset()
                        ret = 0
                        frames = []
                        infos = []
                        for i, a in enumerate(episode['acs']):
                            ob, r, d, info = env.step(a)
                            if args['display'] == 'game':
                                rend = unwrapped_env.render(mode="rgb_array")
                            else:
                                rend = np.asarray(ob)[:, :, :1]
                            frames.append(rend)
                            ret += r
                            infos.append(info)
                            assert not d or i == len(episode['acs']) - 1, ipdb.set_trace()
                        assert d, ipdb.set_trace()
                        assert ret == episode['info']['r'], (ret, episode['info']['r'])
                        episode['obs'] = frames
                        episode['infos'] = infos
                        print(episode.keys())
                        yield episode


class Animation(object):
    def __init__(self, episodes):
        self.episodes = episodes

        self.pause = False
        self.delta = 1
        self.j = 0

        self.fig = self.create_empty_figure()

        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)

        self.axes = {}
        self.lines = {}
        self.dots = {}
        # self.ax1 = self.fig.add_subplot(1, 2, 1)
        # self.ax2 = self.fig.add_subplot(1, 2, 2)

    def create_empty_figure(self):
        fig = plt.figure()
        for evt, callback in fig.canvas.callbacks.callbacks.items():
            items = list(callback.items())
            for cid, _ in items:
                fig.canvas.mpl_disconnect(cid)
        return fig

    def onKeyPress(self, event):
        if event.key == 'left':
            self.pause = True
            if self.j > 0:
                self.j -= 1
        elif event.key == 'right':
            self.pause = True
            if self.j < len(self.episode['obs']) - 1:
                self.j += 1
        elif event.key == 'n':
            self.pause = False
            self.j = len(self.episode['obs']) - 1
        elif event.key == ' ':
            self.pause = not self.pause
        elif event.key == 'q':
            sys.exit()
        elif event.key == 'f':
            self.delta = 1 if self.delta > 1 else 8
        elif event.key == 'b':
            self.j = max(self.j-100, 0)

    def create_axes(self, episode):
        assert self.axes == {}
        keys = [key for key in episode.keys() if key not in ['acs', 'infos', 'obs', 'info']]
        keys.insert(0, 'obs')
        n_rows = int(np.floor(np.sqrt(len(keys))))
        n_cols = int(np.ceil(len(keys) / n_rows))
        for i, key in enumerate(keys, start=1):
            self.axes[key] = self.fig.add_subplot(n_rows, n_cols, i)

    def process_frame(self, frame):
        if frame.shape[-1] == 3:
            return frame
        else:
            return frame[:, :, -1]

    def run(self):
        self.episode = next(self.episodes)

        if self.axes == {}:
            self.create_axes(self.episode)

            self.im = self.axes['obs'].imshow(self.process_frame(self.episode['obs'][0]), cmap='gray')
            for key in self.axes:
                if key != 'obs':
                    line, = self.axes[key].plot(self.episode[key], alpha=0.5)
                    dot = matplotlib.patches.Ellipse(xy=(0, 0), width=1, height=0.0001, color='r')
                    self.axes[key].add_artist(dot)
                    self.axes[key].set_title(key)
                    self.lines[key] = line
                    self.dots[key] = dot


        def draw_frame_i(i):
            # update the data
            if self.j == 0:
                for key in self.axes:
                    if key != 'obs':
                        data = self.episode[key]
                        n_timesteps = len(data)
                        self.lines[key].set_data(range(n_timesteps), data)
                        self.axes[key].set_xlim(0, n_timesteps)
                        min_y, max_y = np.min(data), np.max(data)
                        self.axes[key].set_ylim(min_y, max_y)

                        self.dots[key].height = (max_y - min_y) / 30.
                        self.dots[key].width = n_timesteps / 30.
            self.im.set_data(self.process_frame(self.episode['obs'][self.j]))
            for key in self.axes:
                if key != 'obs':
                    self.dots[key].center = (self.j, self.episode[key][self.j])
            if not self.pause:
                self.j += self.delta
            if self.j > len(self.episode['obs']) - 1:
                self.episode = next(episodes)
                self.j = 0
            return [self.im] + list(self.lines.values()) + list(self.dots.values())

        ani = animation.FuncAnimation(self.fig, draw_frame_i, blit=False, interval=1,
                                      repeat=False)
        plt.show()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_env_params(parser)
    parser.add_argument('--filter', type=str, default='none')
    parser.add_argument('--rew_min', type=int, default=0)
    parser.add_argument('--rew_max', type=int, default=np.inf)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--kind', type=str, default='plot')
    parser.add_argument('--display', type=str, default='game', choices=['game', 'agent'])
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--room_number', type=lambda x: [int(_) for _ in x.split(',')], default=[15])



    args = parser.parse_args().__dict__
    folder = exptag.get_last_experiment_folder_by_tag(args['tag'])

    def date_from_folder(folder):
        assert folder.startswith('openai-')
        date_started = folder[len('openai-'):]
        return datetime.datetime.strptime(date_started, "%Y-%m-%d-%H-%M-%S-%f")

    date_started = date_from_folder(os.path.basename(folder))
    machine_dir = os.path.dirname(folder)
    if machine_dir[-4:-1]=='-00':
        all_machine_dirs = glob.glob(machine_dir[:-1]+'*')
    else:
        all_machine_dirs = [machine_dir]
    other_folders = []
    for machine_dir in all_machine_dirs:
        this_machine_other_folders = os.listdir(machine_dir)
        this_machine_other_folders = [f_ for f_ in this_machine_other_folders
                                      if f_.startswith("openai-") and abs((date_from_folder(f_) - date_started).total_seconds()) < 3]
        this_machine_other_folders = [os.path.join(machine_dir, f_) for f_ in this_machine_other_folders]
        other_folders.extend(this_machine_other_folders)

    filenames = [glob.glob(os.path.join(f_, "videos_*.pk")) for f_ in other_folders]
    assert all(len(files_) == 1 for files_ in filenames), filenames
    filenames = [files_[0] for files_ in filenames]

    env = make_atari(args['env'], max_episode_steps=args['max_episode_steps'])
    if args['display'] == 'agent':
        env = wrap_deepmind(env, frame_stack=4, clip_rewards=False)
    env.reset()
    un_env = env.unwrapped
    rend_shape = un_env.render(mode='rgb_array').shape
    episodes = EpisodeIterator(filenames).iterate()
    if args['kind'] == 'movie':
        import imageio
        import time
        for i, episode in enumerate(episodes):
            filename = os.path.expanduser('~/tmp/movie_{}.mp4'.format(time.time()))
            imageio.mimwrite(filename, episode["obs"], fps=30)
            print(filename)

    else:
        import matplotlib.patches
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        print('left/right, space, n, q, f keys are special')
        Animation(episodes).run()
