"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import sentencepiece as sp


class Tokenizer(object):
    def __init__(self, path):
        self.spp = sp.SentencePieceProcessor()
        self.spp.Load(path)
        self.vocab = [self.spp.IdToPiece(i) for i in range(len(self))]

    def __len__(self):
        return len(self.spp)

    def __call__(self, sent, **kwargs):
        if 'l' in kwargs and 'alpha' in kwargs:
            return self.spp.SampleEncodeAsIds(sent, kwargs['l'], kwargs['alpha'])
        elif 'n' in kwargs:
            return self.spp.NBestEncodeAsIds(sent, kwargs['n'])
        else:
            return self.spp.EncodeAsIds(sent)

    def is_stochastic(self):
        return len(self.spp.SampleEncodeAsIds('test', -1, 0.2)) > 0

    def get_extension(self):
        if not hasattr(self, 'extension'):
            extension = {}
            for i, token in enumerate(self.vocab[3:], 3):
                for l in range(1, len(token)):
                    prefix = token[:l]
                    if prefix not in extension:
                        extension[prefix] = []
                    extension[prefix].append(i)
            self.extension = extension
        return self.extension
