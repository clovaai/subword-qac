"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
import json
import time
import math
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, DataLoader

from tokenization import Tokenizer
from dataset import read_data, PrefixDataset, gen_collate_fn
from metric import calc_rank, calc_partial_rank, mrr_summary, mrl_summary
from utils import model_load


logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
gen_logger = logging.getLogger('generation')
gen_logger.propagate = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description="Generating completions from query prefixes using a language model")
    # data, model directory
    parser.add_argument('--data_dir', default="data/aol/full")
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--min_prefix_len', type=int, default=2)
    parser.add_argument('--min_suffix_len', type=int, default=1)

    # tokenization
    parser.add_argument('--spm', type=str, default='char')

    # evaluation metric
    parser.add_argument('--calc_mrl', action='store_true')

    # test
    parser.add_argument('--n_queries', type=int, default=None)
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--beam_size', type=int, default=30)
    parser.add_argument('--branching_factor', type=int, default=30)
    parser.add_argument('--n_candidates', type=int, default=10)
    parser.add_argument('--retrace', type=int, default=0)
    parser.add_argument('--nbest', type=int, default=1)
    parser.add_argument('--do_merge', action='store_true')
    parser.add_argument('--max_suffix_len', type=int, default=100)

    parser.add_argument('--verbose_completion', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    logger.info(f"device: {device}")

    args.min_len = args.min_prefix_len + args.min_suffix_len
    args.nbest = min(args.nbest, args.beam_size - 1)
    args.branching_factor = min(args.branching_factor, args.beam_size)

    decode_str = (f"+R{args.retrace}" if args.retrace != 0 else "") + ("+M" if args.do_merge else "")
    args.output_dir = args.output_dir or os.path.join('outputs', args.spm, decode_str)
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "generated.txt"), 'w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    gen_logger.addHandler(file_handler)
    gen_logger.propagate = True
    return args


def log_sum_exp(a, b):
    return max(a, b) + np.log(1 + math.exp(-abs(a - b)))


def merge(candidates):
    merged = []
    for candidate, logp in sorted(candidates, key=lambda x: x[0]):
        if len(merged) > 0 and merged[-1][0] == candidate:
            merged[-1] = (candidate, log_sum_exp(merged[-1][1], logp))
        else:
            merged.append((candidate, logp))
    return merged


def remove_duplicates(candidates, prefix, n_candidates, do_merge=False):
    candidates = [(' '.join(candidate.split()), logp) for candidate, logp in candidates]
    candidates = [(candidate, logp) for candidate, logp in candidates if candidate != prefix]

    if do_merge:
        candidates = merge(candidates)

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    if do_merge:
        return [candidate for candidate, _ in candidates[:n_candidates]]
    filtered = []
    for candidate, logp in candidates:
        if candidate not in filtered:
            filtered.append(candidate)
            if len(filtered) == n_candidates:
                break
    return filtered


def beam_search(model, hidden, input, best_score, off, beam_size, branching_factor, max_suffix_len):
    bsz = best_score.size(0)
    batch_idx = torch.arange(bsz).to(device)

    prev_beam_idxs = []
    new_token_idxs = []
    end_scores = []
    end_prev_beam_idxs = []

    for i in range(max_suffix_len):
        output, hidden = model(input, hidden=hidden)            # output: (1, batch * beam, ntoken)
        logp = F.log_softmax(output.squeeze(0), 1)              # logp: (batch * beam, t)
        if i == 0 and off is not None:
            logp.masked_fill_(off.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, -1), -float('inf'))
        score = logp + best_score.view(-1).unsqueeze(1)     # score: (batch * beam, t)

        end_score = score[:, 2].view(-1, beam_size)
        prev_end_score = end_scores[-1] if i > 0 else \
            torch.zeros((bsz, beam_size), dtype=torch.float).fill_(-float('inf')).to(device)
        end_score, end_prev_beam_idx = torch.cat((end_score, prev_end_score), 1).sort(-1, descending=True)
        end_score = end_score[:,:beam_size]                     # end_score: (batch, beam)
        end_prev_beam_idx = end_prev_beam_idx[:, :beam_size]    # end_prev_beam_idx: (batch, beam)
        end_scores.append(end_score)
        end_prev_beam_idxs.append(end_prev_beam_idx)
        score[:, 2].fill_(-float('inf'))

        val, idx0 = score.topk(branching_factor, 1)             # (batch * beam, f)
        val = val.view(-1, beam_size * branching_factor)        # (batch, beam * f)
        idx0 = idx0.view(-1, beam_size * branching_factor)      # (batch, beam * f)
        best_score, idx1 = val.topk(beam_size, 1)               # (batch, beam * f) -> (batch, beam)

        prev_beam_idx = idx1 // branching_factor                # (batch, beam)
        new_token_idx = idx0.gather(1, idx1)                    # (batch, beam)
        prev_beam_idxs.append(prev_beam_idx)
        new_token_idxs.append(new_token_idx)
        input = new_token_idx.view(1, -1)
        hidden_idx = (prev_beam_idx + batch_idx.unsqueeze(1).mul(beam_size)).view(-1)
        hidden = [(h.index_select(0, hidden_idx), c.index_select(0, hidden_idx)) for h, c in hidden]

        if (best_score[:, 0] < end_score[:, -1]).all():
            break

    max_suffix_len = i + 1
    tokens = torch.ones(bsz, beam_size, max_suffix_len, dtype=torch.long).to(device).mul(2) # tokens: (batch, beam, L)
    pos = (beam_size + torch.arange(beam_size)).unsqueeze(0).repeat(bsz, 1).to(device)      # pos: (batch, beam)
    for i in reversed(range(max_suffix_len)):
        end = pos >= beam_size
        for j in range(bsz):
            tokens[j, 1 - end[j], i] = new_token_idxs[i][j, pos[j, 1 - end[j]]]
            pos[j][1 - end[j]] = prev_beam_idxs[i][j, pos[j, 1 - end[j]]]
            pos[j][end[j]] = end_prev_beam_idxs[i][j, pos[j, end[j]] - beam_size]
    decode_len = (tokens != 2).sum(2).max(1)[0]
    return tokens, end_scores[-1], decode_len


def complete(model, tokenizer, batch, args):
    queries, prefixes, previous, target, input, length, mask, off, retrace_idx, nbest_idx = batch

    bsz = len(queries)
    r_bsz = len(prefixes)
    nb_bsz = previous.size(1)
    beam_size = args.beam_size
    branching_factor = args.branching_factor
    max_suffix_len = args.max_suffix_len

    output, hidden = model(previous, length=length)
    raw_loss = F.cross_entropy(output.view(-1, args.ntoken), target.view(-1), reduction='none').view(-1, nb_bsz)
    logp = -(raw_loss * mask.float()).sum(0)

    new_hidden = [(h[:r_bsz].unsqueeze(1).repeat(1, beam_size, 1),
                   c[:r_bsz].unsqueeze(1).repeat(1, beam_size, 1)) for h, c in hidden]
    best_scores = logp.unsqueeze(1).repeat(1, beam_size)  # (batch, beam)
    best_scores[:, 1:].fill_(-float('inf'))
    if args.nbest > 1:
        for i, (s, e) in enumerate(nbest_idx):
            for (nh, nc), (h, c) in zip(new_hidden, hidden):
                nh[i, :-(e - s)] = h[s:e]
                nc[i, :-(e - s)] = c[s:e]
            best_scores[i, :-(e - s)] = logp[s:e]
    hidden = [(h.view(r_bsz * beam_size, -1), c.view(r_bsz * beam_size, -1)) for h, c in new_hidden]

    bs_output = beam_search(model, hidden, input, best_scores, off, beam_size, branching_factor, max_suffix_len)
    tokens, scores, decode_length = [v.tolist() for v in bs_output]

    candidates = [[(prefix + ''.join([tokenizer.vocab[x] for x in t if x != 2]).replace('▁', ' '), s)
                   for t, s in zip(tt, ss)] for prefix, tt, ss in zip(prefixes, tokens, scores)]
    if len(retrace_idx) > 0:
        for i, r in reversed(list(enumerate(retrace_idx))):
            candidates[r].extend(candidates[bsz + i])
            decode_length[r] = max(decode_length[r], decode_length[bsz + i])
        candidates[bsz:] = []
        prefixes[bsz:] = []
        decode_length[bsz:] = []
    candidates = [remove_duplicates(c, prefix, args.n_candidates, args.do_merge)
                  for c, prefix in zip(candidates, prefixes)]
    return candidates, decode_length


def generate(model, tokenizer, data, args, seen_set=None, calc_mrl=False):
    model.eval()
    n_queries = len(data)
    query_lengths = np.array([len(q) for q in data])
    bsz = args.bsz
    seen_set = seen_set or set()
    seens = np.array([int(q in seen_set) for q in data])
    to_save = {'query_lengths': query_lengths, 'seens': seens}

    dataset = PrefixDataset(data, min_prefix_len=args.min_prefix_len, min_suffix_len=args.min_suffix_len)
    data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=bsz, num_workers=args.num_workers,
                             collate_fn=lambda x: gen_collate_fn(x, tokenizer, args))
    start = time.time()
    ranks = []
    pranks = []
    prefix_lengths = []
    decode_lengths = []
    done_prev = done = 0
    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        queries, prefixes = batch[:2]
        pls = [len(p) for p in prefixes]
        # find completion candidates by beam search decoding
        completions, dls = complete(model, tokenizer, batch, args)
        ranks.extend([calc_rank(q, c) for q, c in zip(queries, completions)])
        pranks.extend([calc_partial_rank(q, c) for q, c in zip(queries, completions)])
        prefix_lengths.extend(pls)
        decode_lengths.extend(dls)

        if args.verbose_completion:
            for p, q, c in zip(prefixes, queries, completions):
                gen_logger.info(f"prefix : {p}")
                gen_logger.info('-' * 90)
                gen_logger.info(f"truth  : {q}")
                gen_logger.info('-' * 90)
                for i, x in enumerate(c[:args.n_candidates], 1):
                    gen_logger.info(f"pred{i:02d}{'*' if x == q else ('.' if q.startswith(x + ' ') else ' ')}: {x}")
                gen_logger.info('=' * 90)
                gen_logger.info(" ")

        done += len(queries)
        if done_prev // 10000 < done // 10000:
            logger.info(f"[{done}/{len(data)}]")
            done_prev = done
    mrr_et = time.time() - start
    prefix_lengths = np.array(prefix_lengths)
    decode_lengths = np.array(decode_lengths)
    mdl = decode_lengths.mean()
    gen_logger.info(f"  mean decode length: {mdl:4.1f}")

    qps = n_queries * 1. / mrr_et
    gen_logger.info(f"{mrr_et:4.1f} s | {1000. / qps:4.1f} ms/query | {qps:4.1f} qps")

    mrr_logs = mrr_summary(ranks, pranks, seens, args.n_candidates)
    for log in mrr_logs:
        gen_logger.info(log)
    to_save.update({'prefix_lengths': prefix_lengths, 'decode_lengths': decode_lengths, 
                    'ranks': ranks, 'pranks': pranks})

    if calc_mrl:
        start = time.time()
        remain_idx = np.arange(n_queries)
        recover_lengths = np.zeros((n_queries, args.n_candidates + 1), dtype=np.int)
        last_rank = np.ones(n_queries, dtype=np.int)
        suffix_len = 0
        while len(remain_idx) > 0:
            suffix_len += 1
            logger.info(f"  Processing {len(remain_idx):8d} queries for recover length {suffix_len:2d}")
            dataset = PrefixDataset([data[i] for i in remain_idx], suffix_len=suffix_len)
            data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=bsz,
                                     num_workers=args.num_workers,
                                     collate_fn=lambda x: gen_collate_fn(x, tokenizer, args))

            filtered_idx = []
            for b, batch in enumerate(data_loader):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                part_idx = remain_idx[b * bsz: (b + 1) * bsz]
                queries, prefixes = batch[:2]
                completions, _ = complete(model, tokenizer, batch, args)
                r = np.array([calc_rank(q, c) for q, c in zip(queries, completions)])
                filtered_part_idx = part_idx[r > 0]
                last_rank[filtered_part_idx] = np.maximum(last_rank[filtered_part_idx], r[r > 0])
                for i in filtered_part_idx:
                    recover_lengths[i, last_rank[i]:] += 1
                long_enough = args.min_prefix_len + (suffix_len + 1) <= query_lengths[filtered_part_idx]
                filtered_part_idx = filtered_part_idx[long_enough]
                filtered_idx.append(filtered_part_idx)
            remain_idx = np.concatenate(filtered_idx)
        mrl_et = time.time() - start
        gen_logger.info(f"{mrl_et:6.2f} s")

        mrl_logs = mrl_summary(recover_lengths, seens, args.n_candidates)
        for log in mrl_logs:
            gen_logger.info(log)
        to_save.update({'recover_lengths': recover_lengths})

    if hasattr(args, 'output_dir'):
        torch.save(to_save, open(os.path.join(args.output_dir, "stats.pt"), 'wb'))


def main(args):
    logger.info(f"Args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    spm_path = os.path.join('spm', args.spm, "spm.model")
    logger.info(f"Loading tokenizer from {spm_path}")
    tokenizer = Tokenizer(spm_path)
    args.ntoken = ntoken = len(tokenizer)
    args.branching_factor = min([args.branching_factor, args.ntoken])
    logger.info(f"  Vocab size: {ntoken}")

    n_queries_str = f"{f'only {args.n_queries} samples' if args.n_queries else 'all'} quries from"
    logger.info(f"Reading a dataset ({n_queries_str} test.query.txt)")
    seen_set = set(read_data(os.path.join(args.data_dir, "train.query.txt"), min_len=args.min_len))
    test_data = read_data(os.path.join(args.data_dir, "test.query.txt"), min_len=args.min_len)
    if args.n_queries:
        random.seed(args.seed)
        test_data = random.sample(test_data, args.n_queries)
    n_seen_test_data = len([x for x in test_data if x in seen_set])
    n_unseen_test_data = len(test_data) - n_seen_test_data
    logger.info(f"  Number of test data: {len(test_data):8d} (seen {n_seen_test_data}, unseen {n_unseen_test_data})")

    logger.info(f"Loading model from {args.model_dir}")
    model = model_load(args.model_dir)
    model = model.to(device)

    logger.info('Generation starts!')
    with torch.no_grad():
        generate(model, tokenizer, test_data, args, seen_set=seen_set, calc_mrl=args.calc_mrl)


if __name__ == "__main__":
    main(get_args())
