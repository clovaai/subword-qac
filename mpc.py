"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
import sys
import json
import logging
import argparse
import pickle

from tqdm import tqdm

from dataset import read_data, PrefixDataset
from trie import Trie
from metric import calc_rank, calc_partial_rank, mrr_summary, mrl_summary


logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Most Popular Completion")
    parser.add_argument('--data_dir', default="data/aol/full")
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--min_prefix_len', type=int, default=2)
    parser.add_argument('--min_suffix_len', type=int, default=1)
    parser.add_argument('--n_candidates', type=int, default=10)
    parser.add_argument('--min_freq', type=int, default=1)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model_path', default="models/mpc/trie.pkl")
    args = parser.parse_args()

    return args


def main(args):
    logger.info(f"Args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    logger.info("Reading train dataset")
    train_data = read_data(os.path.join(args.data_dir, f"train.query.txt"), min_len=args.min_len)
    logger.info(f"  Number of train data: {len(train_data):8d}")
    seen_set = set(train_data)

    if not args.train and os.path.isfile(args.model_path):
        logger.info(f"Loading trie at {args.model_path}")
        trie = pickle.load(open(args.model_path, 'rb'))
    else:
        logger.info("Making trie")
        trie = Trie(train_data)

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        logger.info(f"Saving trie at {args.model_path}")
        sys.setrecursionlimit(100000)
        pickle.dump(trie, open(args.model_path, 'wb'))

    logger.info("Reading test dataset")
    test_data = read_data(os.path.join(args.data_dir, f"test.query.txt"), min_len=args.min_len)
    logger.info(f"  Number of  test data: {len(test_data):8d}")

    logger.info("Evaluating MPC")
    test_dataset = PrefixDataset(test_data, args.min_prefix_len, args.min_suffix_len)
    seens = []
    ranks = []
    pranks = []
    rls = []
    for query, prefix in tqdm(test_dataset):
        seen = int(query in seen_set)
        completions = trie.get_mpc(prefix, n_candidates=args.n_candidates, min_freq=args.min_freq)
        rank = calc_rank(query, completions)
        prank = calc_partial_rank(query, completions)
        rl = [0 for _ in range(args.n_candidates + 1)]
        if seen:
            for i in range(1, len(query) + 1):
                r = calc_rank(query, trie.get_mpc(query[:-i]))
                if r == 0:
                    break
                else:
                    for j in range(r, args.n_candidates + 1):
                        rl[j] += 1

        seens.append(seen)
        ranks.append(rank)
        pranks.append(prank)
        rls.append(rl)

    mrr_logs = mrr_summary(ranks, pranks, seens, args.n_candidates)
    mrl_logs = mrl_summary(rls, seens, args.n_candidates)
    for log in mrr_logs + mrl_logs:
        logger.info(log)


if __name__ == "__main__":
    main(get_args())
