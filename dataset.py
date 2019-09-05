"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import hashlib
import unicodedata

import torch
from torch.utils.data import Dataset


def normalizer(query):
    query = unicodedata.normalize('NFKC', query).encode('ascii', 'ignore').decode('ascii')
    query = query.lower()
    query = ' '.join(query.split())
    return query


def read_data(path, min_len=3):
    queries = []
    with open(path) as f:
        for x in f:
            x = x.rstrip('\n')
            if len(x) >= min_len:
                queries.append(x)
    return queries


class QueryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_prefix_len(query, min_prefix_len, min_suffix_len, n=0):
    hasher = hashlib.md5()
    hasher.update(query.encode('utf-8'))
    hasher.update(str(n).encode('utf-8'))
    prefix_len = min_prefix_len + int(hasher.hexdigest(), 16) % (len(query) - min_prefix_len - min_suffix_len + 1)
    # if query[prefix_len - 1] == ' ': prefix_len -= 1
    return prefix_len


class PrefixDataset(Dataset):
    def __init__(self, data, min_prefix_len=2, min_suffix_len=1, suffix_len=None):
        self.data = data
        self.min_prefix_len = min_prefix_len
        self.min_suffix_len = min_suffix_len
        self.suffix_len = suffix_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data[idx]
        pos = -self.suffix_len if self.suffix_len else get_prefix_len(query, self.min_prefix_len, self.min_suffix_len)
        prefix = query[:pos]
        return query, prefix


def collate_fn(queries, tokenizer, sample, max_seq_len=None):
    token_id_seqs = [[1] + tokenizer(x, **sample) + [2] for x in queries]

    length = [len(x) - 1 for x in token_id_seqs]
    if max_seq_len is None or max_seq_len > max(length) + 1:
        max_seq_len = max(length) + 1

    padded = []
    mask = []
    for x in token_id_seqs:
        x = x[:max_seq_len]
        pad_length = max_seq_len - len(x)
        padded.append(x + [0] * pad_length)
        mask.append([1] * (len(x) - 1) + [0] * pad_length)

    padded = torch.tensor(padded).t().contiguous()
    length = torch.tensor(length)
    mask = torch.tensor(mask).t().contiguous()
    return padded[:-1], padded[1:], length, mask


def gen_collate_fn(qps, tokenizer, args):
    queries, prefixes = zip(*qps)

    ntoken = args.ntoken
    beam_size = args.beam_size
    nbest = min(args.nbest, beam_size - 1)
    retrace = args.retrace if hasattr(args, 'retrace') else 0
    is_token_level_subword = any(args.spm.startswith(x) for x in ['bpe', 'unigram'])
    extension = tokenizer.get_extension() if retrace != 0 and is_token_level_subword else None
    bsz = len(prefixes)

    # extend retraced prefixes
    retrace_idx = []
    if retrace:
        retraced_prefixes = []
        off = []
        for i, prefix in enumerate(prefixes):
            e = max(0, len(prefix) - 1 - retrace) if retrace >= 0 else 0
            for s in range(e, len(prefix)):
                inc = prefix[s:]
                if inc in extension:
                    retrace_idx.append(i)
                    retraced_prefixes.append(prefix[:s])
                    o = torch.ones(1, ntoken)
                    o[0, extension[inc]] = 0
                    off.append(o)
        prefixes = list(prefixes) + retraced_prefixes
        off = torch.cat([torch.zeros(bsz, ntoken)] + off, 0).byte()
    else:
        off = None
    r_bsz = len(prefixes)

    # map prefixes to token id sequences
    token_id_seqs = [[1] + tokenizer(x) for x in prefixes]

    # extend for n-best decoding
    nbest_idx = []
    if nbest > 1:
        nb_bsz = r_bsz
        for i, prefix in enumerate(prefixes):
            nlist = [[1] + x for x in tokenizer(prefix, n=nbest)[1:]]
            token_id_seqs.extend(nlist)
            nbest_idx.append((nb_bsz, nb_bsz + len(nlist)))
            nb_bsz += len(nlist)

    length = [len(x) - 1 for x in token_id_seqs]
    max_seq_len = max(length)

    padded = []
    mask = []
    for x in token_id_seqs:
        pad_length = max_seq_len - len(x) + 1
        padded.append(x + [0] * pad_length)
        mask.append([1] * (len(x) - 1) + [0] * pad_length)

    padded = torch.tensor(padded).t().contiguous()
    length = torch.tensor(length)
    mask = torch.tensor(mask).t().contiguous()
    last = torch.tensor([[x[-1]] for x in token_id_seqs])

    if nbest > 1:
        input = last[:r_bsz].repeat(beam_size, 1)
        for i, (s, e) in enumerate(nbest_idx):
            input[i, :-(e - s)] = last[s:e].reshape(-1)
        input = torch.tensor(input).view(1, -1)
    else:
        input = last.t().repeat(beam_size, 1).t().contiguous().view(1, -1)  # (1, batch) -> (1, batch * beam)
    return queries, prefixes, padded[:-1], padded[1:], input, length, mask, off, retrace_idx, nbest_idx
