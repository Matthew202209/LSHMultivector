import os

import numpy as np
import torch
from tqdm import tqdm

from dataloader.colbert_dataloader import ColbertDataset
from encoder.colbert_encoder import ColbertEncoder
from models.base_model import BaseIndex
from utils.colbert_utils import batch


class AllClsIndex(BaseIndex):
    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self):
        corpus_path = r"{}/corpus/{}.jsonl".format(self.config.root_dir, self.config.dataset)
        self.dataset = ColbertDataset(corpus_path)

    def prepare_model(self):
        self.context_encoder = ColbertEncoder(self.config)

    def encode(self):
        self.doclens_list = []
        self.embs_list = []
        for passages_batch in tqdm(batch(self.dataset.corpus_list, self.config.index_batch_size)):
            embs_, doclens_ = self.context_encoder.docFromText(
                passages_batch,
                bsize=self.config.index_batch_size,
                keep_dims="flatten",
                showprogress=(not True),
            )
            
            self.doclens_list += doclens_
            self.embs_list.append(embs_)

        
    def indexing(self):
        self.all_embs = torch.cat(self.embs_list, dim=0)
        self.token_labels = []
        for id, num in enumerate(self.doclens_list):
            for _ in range(num):
                self.token_labels.append(id)
        
        self.token_labels = torch.tensor(self.token_labels).to(torch.int64).to("cpu")
        self.all_embs = self.all_embs.to(torch.float32).to("cpu")

    def indexing(self):
        all_embs = torch.cat(self.embs_list, dim=0)
        self.cls_reps = []
        self.token_reps = []
        self.token_labels = []
        offsets = [0]
        for doclen in self.doclens_list:
            offsets.append(offsets[-1] + doclen)
        i = 0
        n = 0
        d_id = 0
        for token_id, embs in enumerate(tqdm(all_embs)):
            if token_id in offsets:
                cls_rep = embs.unsqueeze(0)
                self.cls_reps.append(cls_rep)
            else:
                token_rep = embs.unsqueeze(0)
                self.token_reps.append(token_rep)
                self.token_labels.append(d_id)
                n += 1
                i += 1
                if n == self.doclens_list[d_id] - 1:
                    n = 0
                    d_id += 1
        self.cls_reps = torch.cat(self.cls_reps, dim=0).to(torch.float32).to("cpu")
        self.token_reps = torch.cat(self.token_reps, dim=0).to(torch.float32).to("cpu")
        self.token_labels = torch.tensor(self.token_labels).to(torch.int64).to("cpu")

    def setup(self):
        self.prepare_model()
        self.prepare_data()

    def fit(self):
        self.encode()
        self.indexing()
        self.save_index()

    def save_index(self):
        save_path = r"{}/index/{}/all_cls".format(self.config.save_dir, self.config.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.token_reps, os.path.join(save_path, "all_cls_token_reps.pt"))
        torch.save(self.cls_reps, os.path.join(save_path, "all_cls_cls_reps.pt"))
        torch.save(self.token_labels, os.path.join(save_path, "token_label.pt"))

