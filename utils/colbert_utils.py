import torch
import transformers
import ujson
from tqdm import tqdm
import itertools
import torch.nn as nn

from transformers import AutoTokenizer


def class_factory():
    pretrained_class = r"BertPreTrainedModel"
    model_class = 'BertModel'
    pretrained_class_object = getattr(transformers, pretrained_class)
    model_class_object = getattr(transformers, model_class)
    class HF_ColBERT(pretrained_class_object):
        """
            Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.

            This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
        """
        _keys_to_ignore_on_load_unexpected = [r"cls"]

        def __init__(self, config, colbert_config):
            super().__init__(config)

            self.config = config
            self.dim = colbert_config.dim
            self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)
            setattr(self, self.base_model_prefix, model_class_object(config))
            self.init_weights()

        @property
        def LM(self):
            base_model_prefix = getattr(self, "base_model_prefix")
            return getattr(self, base_model_prefix)


        @classmethod
        def from_pretrained(cls, name_or_path, colbert_config):
            if name_or_path.endswith('.dnn'):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

                obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
                obj.base = base

                return obj

            obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)
            obj.base = name_or_path

            return obj

        @staticmethod
        def raw_tokenizer_from_pretrained(name_or_path):
            if name_or_path.endswith('.dnn'):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

                obj = AutoTokenizer.from_pretrained(base)
                obj.base = base
                return obj
            obj = AutoTokenizer.from_pretrained(name_or_path)
            obj.base = name_or_path
            return obj
    return HF_ColBERT

class Collection:
    def __init__(self, path=None, corpus_dict=None):
        self.path = path
        self.corpus = corpus_dict or self._load_corpus(path)
        self.corpus_list =  list(self.corpus.values())

    def _load_corpus(self, path):
        self.path = path
        corpus = {}
        num_lines = sum(1 for i in open(self.path, 'rb'))
        with open(self.path, encoding='utf-8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    corpus[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    corpus[line.get("doc_id")] = line.get("text")

        return corpus

    def __iter__(self):
        # TODO: If __data isn't there, stream from disk!
        return self.corpus_list.__iter__()

    def __getitem__(self, item):
        # TODO: Load from disk the first time this is called. Unless self.data is already not None.
        return self.corpus_list[item]

    def __len__(self):
        # TODO: Load here too. Basically, let's make data a property function and, on first call, either load or get __data.
        return len(self.corpus_list)

    def enumerate(self, rank = 1):
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank =1, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(1))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return

    def get_chunksize(self):
        return min(25_000, 1 + len(self) // 1)  # 25k is gr

class MixedPrecisionManager():
    def __init__(self, activated):
        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


def torch_load_dnn(path):
    if path.startswith("http:") or path.startswith("https:"):
        dnn = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        dnn = torch.load(path, map_location='cpu')

    return dnn


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return



def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices



def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
