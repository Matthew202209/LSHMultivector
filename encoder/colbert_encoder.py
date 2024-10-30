import argparse
import string

import transformers
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from tokenization.colbert_doc_tokenization import DocTokenizer
from tokenization.colbert_query_tokenization import QueryTokenizer
from utils.colbert_utils import MixedPrecisionManager,class_factory


class ColbertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_gpu =False
        HF_ColBERT = class_factory()
        self.model = HF_ColBERT.from_pretrained(self.config.checkpoints_dir, colbert_config=self.config).to(self.config.device)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoints_dir)

        self.query_tokenizer = QueryTokenizer(self.config)
        self.doc_tokenizer = DocTokenizer(self.config)

        self.amp_manager = MixedPrecisionManager(True)
        self.eval()
        self.skiplist = {w: True
                         for symbol in string.punctuation
                         for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.pad_token = self.raw_tokenizer.pad_token_id
        # self.pad_token = 103




    def _query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def _doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D

    def encode_query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self._query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def encode_doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D= self._doc(*args, **kw_args)
                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False):

        # perf_encode = perf_event.PerfEvent()
        # todo query_tokenizer
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, context=context, bsize=bsize, full_length_search=full_length_search)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries, context=context, full_length_search=full_length_search)

        num_token = torch.sum(attention_mask.squeeze(0)).item()
        query = self.encode_query(input_ids, attention_mask)

        return query[:,:num_token,:]

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, 'flatten']

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
            batches = [(self.encode_doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu), input_ids, attention_mask)
                       for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == 'flatten':
                D, mask, ids = [], [], []

                for (D_, mask_), input_ids, attention_mask in batches:
                    D.append(D_)
                    mask.append(mask_)
                    ids.append(input_ids)

                D, mask, ids = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices], torch.cat(ids)[reverse_indices]

                doclens = mask.squeeze(-1).sum(-1).tolist()
                D = D.view(-1, self.config.dim)
                D = D[mask.bool().flatten()].cpu()
                ids = ids.flatten()[mask.bool().flatten().cpu()].tolist()
                import json
                with open('ids.jsonl', 'at') as fout:
                    json.dump(ids, fout)
                    fout.write('\n')

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)
        #todo doc_tokenizer
        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.encode_doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.LM

    @property
    def linear(self):
        return self.model.linear

    @property
    def score_scaler(self):
        return self.model.score_scaler

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output


