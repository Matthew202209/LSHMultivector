import os
import dill as pickle

import numpy as np
import torch
from tqdm import tqdm

from dataloader.colbert_dataloader import ColbertDataset
from encoder.colbert_encoder import ColbertEncoder
from models.base_model import BaseIndex
from utils.colbert_utils import batch
import iSaxMtree


class IsaxMtreeIndex(BaseIndex):
    def __init__(self, config):
        super().__init__(config)
        self.vector_db = None
        self.cls_reps = []
        self.token_labels = []
        self.token_reps = []
        self.token_reps_expansion = None
        self.token_d_ids = []

    def prepare_data(self):
        corpus_path = r"{}/corpus/{}.jsonl".format(self.config.root_dir, self.config.dataset)
        self.dataset = ColbertDataset(corpus_path)

    def prepare_model(self):
        self.context_encoder = ColbertEncoder(self.config)

    def expansion(self):
        assert self.token_reps is not None
        norm_squares = np.sum(self.token_reps ** 2, axis=1)
        max_norm_square = np.max(norm_squares)
        # 计算每个向量的新维度的值
        new_dimension = np.sqrt(max_norm_square - norm_squares)
        # 将新维度增加一个维度，从(10000,)变为(10000, 1)
        new_dimension = new_dimension[:, np.newaxis]
        random_signs = np.random.choice([-1, 1], size=new_dimension.shape)
        # 将随机向量与new_dimension相乘
        new_dimension = new_dimension * random_signs
        # 将新维度添加到原来的向量中，形成(10000, 129)的数组
        self.token_reps_expansion = np.hstack((self.token_reps, new_dimension))


    def init_isax_mtree_index(self):
        config = iSaxMtree.Config()
        config.num_data = self.token_reps_expansion.shape[0]
        config.highDim = self.token_reps_expansion.shape[1]
        config.lowDim  =self.config.low_dim
        config.save_dir = r"{}/{}/isax_mtree".format(self.config.save_dir, self.config.dataset)
        config.Sample_p =  self.config.sample_p
        config.sax_alphabet_cardinality = self.config.sax_alphabet_cardinality
        config.pivotNum = self.config.pivot_num
        config.pivotRandomCount = self.config.pivot_random_count
        config.c_appro = self.config.c_appro
        config.T = config.num_data * 0.1
        config.alpha1 =  self.config.alpha1
        config.t = iSaxMtree.MyFunc.Ccal_thresh(config.lowDim, config.alpha1)
        config.M_NUM =  self.config.m_num
        config.MLeaf = self.config.mleaf
        self.vector_db = iSaxMtree.VectorDB(config)
    def setup(self):
        self.prepare_data()
        self.prepare_model()

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

    def isax_mtree_create(self):
        self.vector_db.inputDocTokenRepres(self.token_reps_expansion)
        self.vector_db.iSaxProcess()
        self.vector_db.mTreeIndexConstruct()
    def indexing(self):
        assert self.embs_list is not None
        all_embs = torch.cat(self.embs_list, dim=0)
        offsets = [0]
        for doclen in self.doclens_list:
            offsets.append(offsets[-1] + doclen)
        i = 0
        n = 0
        d_id = 0
        for token_id, embs in enumerate(tqdm(all_embs)):
            if token_id in offsets:
                cls_rep = embs.unsqueeze(0).detach().numpy()
                self.cls_reps.append(cls_rep)
            else:
                token_rep = embs.unsqueeze(0).detach().numpy()
                self.token_reps.append(token_rep)
                self.token_d_ids.append(d_id)
                n +=1
                i +=1
                if n == self.doclens_list[d_id] - 1:
                    n = 0
                    d_id += 1

        self.cls_reps = np.concatenate(self.cls_reps, axis=0)
        self.token_reps = np.concatenate(self.token_reps, axis=0)
        self.token_d_ids = np.array(self.token_d_ids)
        self.save_index()
        self.expansion()
        self.init_isax_mtree_index()
        self.isax_mtree_create()


    def save_index(self):
        save_path = r"{}/{}/isax_mtree".format(self.config.save_dir, self.config.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "cls_reps.npy"), self.cls_reps)
        np.save(os.path.join(save_path, "token_reps.npy"), self.token_reps)
        np.save(os.path.join(save_path, "token_d_ids.npy"), self.token_d_ids)
