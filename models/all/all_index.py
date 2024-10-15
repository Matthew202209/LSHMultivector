import os

import numpy as np
import torch
from tqdm import tqdm

from dataloader.colbert_dataloader import ColbertDataset
from encoder.colbert_encoder import ColbertEncoder
from models.base_model import BaseIndex
from utils.colbert_utils import batch


class AllIndex(BaseIndex):
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
            

    def setup(self):
        self.prepare_model()
        self.prepare_data()

    def fit(self):
        self.encode()
        self.indexing()
        self.save_index()

    def save_index(self):
        save_path = r"{}/index/{}/all".format(self.config.save_dir, self.config.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.all_embs, os.path.join(save_path, "all_embs.pt"))
        torch.save(self.token_labels, os.path.join(save_path, "token_label.pt"))

