"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
import argparse
import datetime

from tqdm import tqdm

from dataset import normalizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='full')
    parser.add_argument('--train_start', type=str, default='2006-03-01 00:00:00')
    parser.add_argument('--train_end',   type=str, default='2006-05-18 00:00:00')
    parser.add_argument('--valid_start', type=str, default='2006-05-18 00:00:00')
    parser.add_argument('--valid_end',   type=str, default='2006-05-25 00:00:00')
    parser.add_argument('--test_start',  type=str, default='2006-05-25 00:00:00')
    parser.add_argument('--test_end',    type=str, default='2006-06-01 00:00:00')
    args = parser.parse_args()
    return args


def main(args):
    splits = ['train', 'valid', 'test']
    columns = ['uid', 'query', 'time']
    fmt = '%Y-%m-%d %H:%M:%S'

    print(f"Split original data into data/aol/{args.tag}")
    itv = {s: tuple(vars(args)[f"{s}_{i}"] for i in ['start', 'end']) for s in splits}
    for s in splits:
        print(f"  {s:5s} data: from {itv[s][0]} until {itv[s][1]}")
    itv = {k: tuple(datetime.datetime.strptime(x, fmt) for x in v) for k, v in itv.items()}
    valid = (itv['train'][0] < itv['train'][1] <= itv['valid'][0] < itv['valid'][1] <= itv['test'][0] < itv['test'][1])
    assert valid, "Invalid time intervals"

    # make directory and open files to write
    target_dir = f"data/aol/{args.tag}"
    os.makedirs(target_dir, exist_ok=True)
    f = {s: {column: open(os.path.join(target_dir, f"{s}.{column}.txt"), 'w') for column in columns} for s in splits}

    # read original AOL query log dataset and write data into files
    print("")
    cnt = {s: 0 for s in splits}
    for i in range(1, 11):
        filename = f"user-ct-test-collection-{i:02d}.txt"
        print(f"Reading {filename}...")
        f_org = open(os.path.join("data/aol/org", filename))
        f_org.readline()
        prev = {column: '' for column in columns}
        for line in tqdm(f_org):
            data = {column: v for column, v in zip(columns, line.strip().split('\t')[:3])}
            # normalize queries
            data['query'] = normalizer(data['query'])
            # filter out too short queries and redundant queries
            # data['query'] == '-'
            if len(data['query']) < 3 or (data['uid'], data['query']) == (prev['uid'], prev['query']):
                continue
            t = datetime.datetime.strptime(data['time'], fmt)
            for s in splits:
                if itv[s][0] <= t < itv[s][1]:
                    cnt[s] += 1
                    for column in columns:
                        f[s][column].write(data[column] + '\n')
            prev = data

    # print total number of data in each split
    print("")
    for s in splits:
        print(f"Number of {s:5s} data: {cnt[s]:8d}")


if __name__ == "__main__":
    main(get_args())
