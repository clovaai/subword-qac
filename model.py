"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import json
import copy
import math

import torch
import torch.nn as nn
from torch.nn import Parameter


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, layernorm=False, dropoutr=0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = Parameter(torch.Tensor(input_size + hidden_size, 3 * hidden_size))

        self.bias = Parameter(torch.Tensor(3 * hidden_size)) if bias else None
        self.layernorm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(3)]) if layernorm else None
        self.dropoutr = nn.Dropout(dropoutr) if dropoutr > 0 else None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.layernorm:
            for ln in self.layernorm:
                ln.reset_parameters()

    def forward(self, input, hx=None):
        # input: (batch, input_size)
        # hx: tuple of (batch, hidden_size), (batch, hidden_size)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        h_0, c_0 = hx

        pre_gate = torch.mm(torch.cat((input, h_0), 1), self.weight)
        if self.bias: pre_gate += self.bias.expand(0)
        f, o, g = pre_gate.split(self.hidden_size, 1)
        if self.layernorm:
            f = self.layernorm[0](f)
            o = self.layernorm[1](o)
            g = self.layernorm[2](g)
        f = torch.sigmoid(f + 1.)   # _forget_bias
        i = 1. - f                  # input and forget gates are coupled
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        if self.dropoutr:
            g = self.dropoutr(g)    # recurrent dropout without memory loss

        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        return h_1, c_1


class LSTM(nn.Module):
    def __init__(self, dims, bias=True, layernorm=False, dropoutr=0, dropouth=0, dropouto=0, batch_first=False):
        super(LSTM, self).__init__()
        self.dims = dims
        self.nlayers = len(self.dims) - 1
        self.bias = bias
        self.layernorm = layernorm
        self.dropouth = nn.Dropout(dropouth) if dropouth > 0 else None
        self.dropouto = nn.Dropout(dropouto) if dropouto > 0 else None
        self.batch_first = batch_first
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size, bias=bias, layernorm=layernorm, dropoutr=dropoutr)
                      for input_size, hidden_size in zip(dims[:-1], dims[1:])])
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return [(weight.new(batch_size, hidden_size).zero_(),
                 weight.new(batch_size, hidden_size).zero_())
                for hidden_size in self.dims[1:]]

    @staticmethod
    def _forward_unroll(cell, input, hx, length=None):
        seq_len = input.size(0)
        num_continues = input.size(1)
        output = []
        for time in range(seq_len):
            if length is not None:
                while length[num_continues - 1] <= time:
                    num_continues -= 1
            h_next, c_next = cell(input[time, :num_continues], (hx[0][:num_continues], hx[1][:num_continues]))
            h_next = torch.cat((h_next[:num_continues], hx[0][num_continues:]), 0)
            c_next = torch.cat((c_next[:num_continues], hx[1][num_continues:]), 0)
            output.append(h_next)
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx   # output: (seq_len, batch, d_{i+1})

    def forward(self, input, hxs=None, length=None):
        # input: (seq_len, batch, d_0)
        # hxs: list of (h_i, c_i) with length layer, where h_i, c_i : (batch, d_i)
        # length: (batch)
        if self.batch_first:
            input = input.transpose(0, 1)

        if length is not None:
            length, perm_idx = length.sort(0, descending=True)
            input = input[:, perm_idx]
            hxs = [(h[perm_idx], c[perm_idx]) for h, c in hxs] if hxs else None

        if not hxs: hxs = self.init_hidden(input.size(1))
        new_hxs = []
        for l, (cell, hx) in enumerate(zip(self.cells, hxs)):
            output, new_hx = self._forward_unroll(cell, input, hx, length)
            new_hxs.append(new_hx)
            if l != self.nlayers - 1 and self.dropouth:
                output = self.dropouth(output)
            input = output
        if self.dropouto:
            output = self.dropouto(output)

        if length is not None:
            inv_perm_idx = perm_idx.sort(0)[1]
            output = output[:, inv_perm_idx]
            new_hxs = [(h[inv_perm_idx], c[inv_perm_idx]) for h, c in new_hxs]
        return output, new_hxs  # output: (seq_len, batch, d_{-1})


class LMConfig(object):
    def __init__(self, ntoken_or_config_json_file, ninp=100, nhid=600, nlayers=1,
                 dropouti=0, dropoutr=0.25, dropouth=0, dropouto=0):
        if isinstance(ntoken_or_config_json_file, str):
            with open(ntoken_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            self.ntoken = ntoken_or_config_json_file
            self.ninp = ninp
            self.nhid = nhid
            self.nlayers = nlayers
            self.dropouti = dropouti
            self.dropoutr = dropoutr
            self.dropouth = dropouth
            self.dropouto = dropouto

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config
        self.ntoken = ntoken = config.ntoken
        self.ninp = ninp = config.ninp
        self.nhid = nhid = config.nhid
        self.nlayers = nlayers = config.nlayers

        self.encoder = nn.Embedding(ntoken, ninp)
        self.dropouti = nn.Dropout(config.dropouti) if config.dropouti > 0 else None
        self.lstm = LSTM([ninp] + [nhid] * nlayers, bias=False, layernorm=True,
                         dropoutr=config.dropoutr, dropouth=config.dropouth, dropouto=config.dropouto)
        self.projection = nn.Linear(nhid, ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden=None, length=None):
        emb = self.encoder(input)
        if self.dropouti:
            emb = self.dropouti(emb)
        if length is not None and len(length.size()) > 1:
            length = length.squeeze(0)
        output, hidden = self.lstm(emb, hidden, length)
        output = self.projection(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return result, hidden
