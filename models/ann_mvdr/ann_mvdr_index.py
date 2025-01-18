import os
import dill as pickle
import faiss

import numpy as np
import torch
from numpy.distutils.command.config import config
from tqdm import tqdm

from dataloader.colbert_dataloader import ColbertDataset
from encoder.colbert_encoder import ColbertEncoder
from models.base_model import BaseIndex
from utils.colbert_utils import batch


class AnnMvdrIndex(BaseIndex):
    def __init__(self, config):
        super().__init__(config)
        self.cls_reps = []
        self.token_d_ids = []
        self.token_reps = []
        self.faiss_db = None

    def prepare_data(self):
        corpus_path = r"{}/corpus/{}.jsonl".format(self.config.root_dir, self.config.dataset)
        self.dataset = ColbertDataset(corpus_path)

    def prepare_model(self):
        self.context_encoder = ColbertEncoder(self.config)

    def init_faiss_db(self):
        if self.config.ann_type == "IndexLSH":
            assert self.config.n_bits_lsh
            self.faiss_db = faiss.IndexLSH(self.config.dim, self.config.n_bits_lsh)
        elif self.config.ann_type == "IndexHNSW":
            assert self.config.max_layer
            assert self.config.num_neighbor
            self.faiss_db = faiss.IndexHNSWFlat(self.config.dim, self.config.max_layer)  # 32
            self.faiss_db.hnsw.efConstruction = self.config.num_neighbor
        elif self.config.ann_type == "IndexIVFPQ":
            #  self.config.m 子量化器的数量
            # self.config.nlist 聚类中心的数量
            #  self.config.n_bits_fpq 每个码字的位数
            # self.config.m 子量化器的数量
            quantizer = faiss.IndexFlatL2(self.config.dim)
            self.faiss_db = faiss.IndexIVFPQ(quantizer, self.config.dim,
                                             self.config.nlist, self.config.m, self.config.n_bits_fpq)

    def setup(self):
        self.prepare_data()
        self.prepare_model()
        self.init_faiss_db()

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
        offsets = [0]
        for doclen in self.doclens_list:
            offsets.append(offsets[-1] + doclen)
        i = 0
        n = 0
        d_id = 0
        if self.config.ann_type == "IndexIVFPQ":
            self.faiss_db.train(self.all_embs.detach().numpy())
        for token_id, embs in enumerate(tqdm(self.all_embs)):
            if token_id in offsets:
                cls_rep = embs.unsqueeze(0).detach().numpy()
                self.cls_reps.append(cls_rep)
            else:
                token_rep = embs.unsqueeze(0).detach().numpy()
                self.token_reps.append(token_rep)
                self.token_d_ids.append(d_id)

                self.faiss_db.add(token_rep)
                n +=1
                i +=1
                if n == self.doclens_list[d_id] - 1:
                    n = 0
                    d_id += 1

        self.cls_reps = np.concatenate(self.cls_reps, axis=0)
        self.token_reps = np.concatenate(self.token_reps, axis=0)
        self.token_d_ids = np.array(self.token_d_ids)

    def save_index(self):
        save_path = r"{}/index/{}/ann_mvdr/{}".format(self.config.save_dir, self.config.dataset, self.config.ann_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "cls_reps.npy"), self.cls_reps)
        np.save(os.path.join(save_path, "token_reps.npy"), self.token_reps)
        np.save(os.path.join(save_path, "token_d_ids.npy"), self.token_d_ids)
        faiss.write_index(self.faiss_db, r"{}/vector_db.index".format(save_path))

