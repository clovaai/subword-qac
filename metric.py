"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import numpy as np


def calc_rank(truth, candidates):
    return 1 + candidates.index(truth) if truth in candidates else 0


def calc_partial_rank(truth, candidates):
    for i, candidate in enumerate(candidates, 1):
        if truth == candidate or truth.startswith(candidate + ' '):
            return i
    return 0


def mrr_summary(ranks, pranks, seens, n_candidates):
    ranks = np.array(ranks)
    pranks = np.array(pranks)

    n = np.zeros(3, dtype=int)
    rank = np.zeros((3, n_candidates + 1), dtype=int)
    prank = np.zeros((3, n_candidates + 1), dtype=int)
    reciprocal = np.array([0.] + [1. / r for r in range(1, n_candidates + 1)]).reshape(1, -1)
    for s, r, pr in zip(seens, ranks, pranks):
        for i in [1 - s, 2]:
            n[i] += 1
            rank[i, r] += 1
            prank[i, pr] += 1
    mrr = np.cumsum(rank * reciprocal, 1) / n.reshape((3, 1))
    pmrr = np.cumsum(prank * reciprocal, 1) / n.reshape((3, 1))
    
    logs = []
    for i in range(1, n_candidates + 1):
        i_str = ' '.join(f"{mrr[s, i]:.4f} ({seen_str})" for s, seen_str in enumerate(['seen', 'unseen', 'all']))
        logs.append(f"mrr @{i:-2d}: {i_str}")
    logs.append(" ")
    for i in range(1, n_candidates + 1):
        i_str = ' '.join(f"{pmrr[s, i]:.4f} ({seen_str})" for s, seen_str in enumerate(['seen', 'unseen', 'all']))
        logs.append(f"pmrr @{i:-2d}: {i_str}")
    logs.append(" ")
    return logs


def mrl_summary(recover_lengths, seens, n_candidates):
    recover_lengths = np.array(recover_lengths)
    seens = np.array(seens)
    mrl = np.concatenate((recover_lengths[seens == 1].mean(0).reshape((1, -1)),
                          recover_lengths[seens == 0].mean(0).reshape((1, -1)),
                          recover_lengths.mean(0).reshape((1, -1))), 0)

    logs = []
    for i in range(1, n_candidates + 1):
        i_str = ' '.join(f"{mrl[s, i]:.4f} ({seen_str})" for s, seen_str in enumerate(['seen', 'unseen', 'all']))
        logs.append(f"mrl @{i:-2d}: {i_str}")
    logs.append(" ")
    return logs
