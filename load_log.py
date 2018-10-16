import argparse
import numpy as np
import os
import pickle
import exptag

separator = '------------'

def parse(key, value):
    value = value.strip()
    try:
        if value.startswith('['):
            value = value[1:-1].split(';')
            if value != ['']:
                try:
                    value = [int(v) for v in value]
                except:
                    value = [str(v) for v in value]
            else:
                value = []
        elif ';' in value:
            value = 0.
        elif value in ['nan', '', '-inf']:
            value = np.nan
        else:
            value = eval(value)
    except:
        import ipdb; ipdb.set_trace()
        print(f"failed to parse value {key}:{value.__repr__()}")
        value = 0.

    return value

def get_hash(filename):
    import hashlib
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def pickle_cache_result(f):
    def cached_f(filename):
        current_hash = get_hash(filename)
        cache_filename = filename + '_cache'
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as fl:
                try:
                    stored_hash, stored_result = pickle.load(fl)
                    if stored_hash == current_hash:
                        # pass
                        return stored_result
                except:
                    pass
        result = f(filename)
        with open(cache_filename, 'wb') as fl:
            pickle.dump((current_hash, result), fl)
        return result
    return cached_f

@pickle_cache_result
def parse_csv(filename):
    import csv

    timeseries = {}
    keys = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                keys = row
            else:
                for column, key in enumerate(keys):
                    if key not in timeseries:
                        timeseries[key] = []
                    timeseries[key].append(parse(key, row[column]))
    print(f'parsing {filename}')
    if 'opt_featvar' in timeseries:
        timeseries['opt_feat_var'] = timeseries['opt_featvar']
    return timeseries


def get_filename_from_tag(tag):
    folder = exptag.get_last_experiment_folder_by_tag(tag)
    return os.path.join(folder, "progress.csv")

def get_filenames_from_tags(tags):
    return [get_filename_from_tag(tag) for tag in tags]

def get_timeseries_from_filenames(filenames):
    return [parse_csv(f) for f in filenames]


def get_timeseries_from_tags(tags):
    return get_timeseries_from_filenames(get_filenames_from_tags(tags))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tags', type=lambda x: x.split(','), nargs='+', default=None)
    parser.add_argument('--x_axis', type=str, choices=['tcount', 'n_updates'], default='tcount')
    args = parser.parse_args()

    x_axis = args.x_axis

    timeseries_groups = []
    for tag_group in args.tags:
        timeseries_groups.append(get_timeseries_from_tags(tags=tag_group))
    for tag_group, timeseries in zip(args.tags, timeseries_groups):
        rooms = []
        for tag, t in zip(tag_group, timeseries):
            if 'rooms' in t:
                rooms.append((tag, t['rooms'][-1], t['best_ret'][-1], t[x_axis][-1]))
            else:
                print(f"tag {tag} has no rooms")
        rooms = sorted(rooms, key=lambda x: len(x[1]))
        all_rooms = set.union(*[set(r[1]) for r in rooms])
        import pprint
        for tag, r, best_ret, max_x in rooms:
            print(f'{tag}:{best_ret}:{r}(@{max_x})')
        pprint.pprint(all_rooms)
    keys = set.intersection(*[set(t.keys()) for t in sum(timeseries_groups, [])])

    keys = sorted(list(keys))
    import matplotlib.pyplot as plt

    n_rows = int(np.ceil(np.sqrt(len(keys))))
    n_cols = len(keys) // n_rows + 1
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True)
    for i in range(n_rows):
        for j in range(n_cols):
            ind = i * n_cols + j
            if ind < len(keys):
                key = keys[ind]
                for color, timeseries in zip('rgbykcm', timeseries_groups):
                    if key in ['all_places', 'recent_places', 'global_all_places', 'global_recent_places', 'rooms']:
                        if any(isinstance(t[key][-1], list) for t in timeseries):
                            for t in timeseries:
                                t[key] = list(map(len, t[key]))
                    max_timesteps = min((len(_[x_axis]) for _ in timeseries))
                    try:
                        data = np.asarray([t[key][:max_timesteps] for t in timeseries], dtype=np.float32)
                    except:
                        import ipdb; ipdb.set_trace()
                    lines = [np.nan_to_num(d[key]) for d in timeseries]
                    lines_x = [np.asarray(d[x_axis]) for d in timeseries]
                    alphas = [0.2/np.sqrt(len(lines)) for l in lines]
                    lines += [np.nan_to_num(np.nanmean(data, 0))]
                    alphas += [1.]
                    lines_x += [np.asarray(timeseries[0][x_axis][:max_timesteps])]
                    for alpha, y, x in zip(alphas, lines, lines_x):
                        axes[i, j].plot(x, y, color=color, alpha=alpha)
                axes[i, j].set_title(key)

    plt.show()
    plt.close()