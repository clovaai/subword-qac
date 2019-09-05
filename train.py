"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
import json
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tokenization import Tokenizer
from dataset import read_data, QueryDataset, collate_fn
from model import LMConfig, LanguageModel
from utils import get_params, get_model, model_save, model_load, TrainLogger


logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


def get_args():
    parser = argparse.ArgumentParser(description="Training a LSTM language model for queries")
    # data, model directory
    parser.add_argument('--data_dir', default="data/aol/full")
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    parser.add_argument('--min_len', type=int, default=3)   # min_prefix_len + min_suffix_len

    # tokenization
    parser.add_argument('--spm', type=str, default='char')
    parser.add_argument('--sample', nargs='*', type=float)

    # model
    parser.add_argument('--ninp', type=int, default=100)
    parser.add_argument('--nhid', type=int, default=600)
    parser.add_argument('--nlayers', type=int, default=1)

    # dropout
    parser.add_argument('--dropouti', type=float, default=0)
    parser.add_argument('--dropoutr', type=float, default=0.25)
    parser.add_argument('--dropouth', type=float, default=0)
    parser.add_argument('--dropouto', type=float, default=0)

    # training
    parser.add_argument('--max_seq_len', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--bsz', type=int, default=1024)
    parser.add_argument('--clip', default=0.25)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=None)
    parser.add_argument('--eval_n_steps', type=int, default=None)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    logger.info(f"device: {device}, n_gpu: {n_gpu}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    args.model_dir = args.model_dir or os.path.join('models', args.spm)
    return args


def parse_sample_options(x):
    opt = {}
    if x:
        if len(x) == 2:
            opt = {'l': int(x[0]), 'alpha': x[1]}
        elif len(x) == 1:
            opt = {'n': int(x[0])}
    return opt


def calc_loss(model, batch):
    previous, target, length, mask = batch
    output, _ = model(previous, length=length.unsqueeze(0))
    bsz = previous.size(1)
    raw_loss = F.cross_entropy(output.view(-1, get_model(model).ntoken), target.view(-1), reduction='none')
    raw_loss = raw_loss.view(-1, bsz)
    loss = (raw_loss * mask.float()).sum(0).mean()
    items = [loss.data.item(), bsz, mask.sum().item()]
    return loss, items


def evaluate(model, data_loader, n_steps=None):
    model.eval()
    train_logger = TrainLogger()
    for step, batch in enumerate(data_loader, 1):
        if n_steps and step > n_steps:
            break
        batch = tuple(t.to(device) for t in batch)
        _, items = calc_loss(model, batch)
        train_logger.add(*items)
    return train_logger.average(), train_logger.print_str()


def train(model, optimizer, tokenizer, train_data, valid_data, args):
    logger.info("Training starts!")
    os.makedirs(args.model_dir, exist_ok=True)

    train_dataset = QueryDataset(train_data)
    train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                   batch_size=args.bsz, num_workers=args.num_workers,
                                   collate_fn=lambda x: collate_fn(x, tokenizer, args.sample, args.max_seq_len))

    valid_dataset = QueryDataset(valid_data)
    valid_data_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                   batch_size = args.bsz, num_workers = args.num_workers,
                                   collate_fn = lambda x: collate_fn(x, tokenizer, args.sample, args.max_seq_len))

    n_batch = (len(train_dataset) - 1) // args.bsz + 1
    logger.info(f"  Number of training batch: {n_batch}")
    if args.eval_interval is None:
        args.eval_interval = n_batch

    try:
        best_valid_loss = float('inf')
        model.train()
        params = get_params(model)
        train_logger = TrainLogger()
        train_logger_part = TrainLogger()
        step = 0
        for epoch in range(1, args.n_epochs + 1):
            logger.info(f"Epoch {epoch:2d}")
            for batch in train_data_loader:
                step += 1
                batch = tuple(t.to(device) for t in batch)
                loss, items = calc_loss(model, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.clip)
                optimizer.step()
                optimizer.zero_grad()
                train_logger.add(*items)
                train_logger_part.add(*items)

                if step % args.log_interval == 0:
                    logger.info(f"  step {step:8d} | {train_logger_part.print_str(True)}")
                    train_logger_part.init()

                if step % args.eval_interval == 0:
                    start_eval = time.time()
                    logger.info('-' * 90)
                    train_loss, train_str = train_logger.average(), train_logger.print_str()
                    logger.info(f"| step {step:8d} | train | {train_str}")

                    # evaluate valid loss, ppl
                    with torch.no_grad():
                        valid_loss, valid_str = evaluate(model, valid_data_loader, args.eval_n_steps)
                    logger.info(f"| step {step:8d} | valid | {valid_str}")
                    if valid_loss[0] < best_valid_loss:
                        model_save(args.model_dir, model, optimizer)
                        logger.info(">>>>> Saving model (new best validation)")
                        best_valid_loss = valid_loss[0]
                    logger.info('-' * 90)

                    model.train()
                    train_logger.init()
                    train_logger_part.start += time.time() - start_eval

    except KeyboardInterrupt:
        logger.info('-' * 90)
        logger.info('  Exiting from training early')


def test(model, tokenizer, test_data, args):
    logger.info("Test starts!")
    model_load(args.model_dir, model)
    model = model.to(device)

    test_dataset = QueryDataset(test_data)
    test_data_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                  batch_size=args.bsz, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, args.sample, args.max_seq_len))

    test_loss, test_str = evaluate(model, test_data_loader)
    logger.info(f"| test  | {test_str}")


def main(args):
    logger.info(f"Args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    spm_path = os.path.join('spm', args.spm, "spm.model")
    args.sample = parse_sample_options(args.sample)
    logger.info(f"Loading tokenizer from {spm_path}")
    tokenizer = Tokenizer(spm_path)
    args.ntoken = ntoken = len(tokenizer)
    logger.info(f"  Vocabulary size: {ntoken}")

    logger.info("Reading dataset")
    data = {}
    for x in ['train', 'valid', 'test']:
        data[x] = read_data(os.path.join(args.data_dir, f"{x}.query.txt"), min_len=args.min_len)
        logger.info(f"  Number of {x:>5s} data: {len(data[x]):8d}")

    logger.info("Preparing model and optimizer")
    config = LMConfig(ntoken, args.ninp, args.nhid, args.nlayers,
                      args.dropouti, args.dropoutr, args.dropouth, args.dropouto)
    model = LanguageModel(config).to(device)
    params = get_params(model)
    logger.info(f"  Number of model parameters: {sum(p.numel() for p in params)}")
    optimizer = torch.optim.Adam(params)

    if args.resume:
        logger.info(f"Loading model from {args.resume}")
        model_load(args.resume, model, optimizer)
        model = model.to(device)

    if n_gpu > 1:
        logger.info(f"Making model as data parallel")
        model = torch.nn.DataParallel(model, dim=1)

    train(model, optimizer, tokenizer, data['train'], data['valid'], args)

    test(model, tokenizer, data['test'], args)


if __name__ == "__main__":
    main(get_args())
