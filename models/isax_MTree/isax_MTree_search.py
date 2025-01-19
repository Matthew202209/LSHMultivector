import os

import iSaxMtree
import numpy as np
import torch
import torch_scatter
from numpy.distutils.command.config import config
from tqdm import tqdm
from bitstring import BitArray
from models.Hamming.hamming_search import HammingSearcher
import iSaxMtree


class  IsaxMTreeSearcher(HammingSearcher):
    def __init__(self, config, num_doc):
        super().__init__(config, num_doc)
        self.token_d_ids = None
        self.token_reps = None
        self.vector_db = None

    def prepare_index(self):
        index_dir = r"{}/{}/isax_mtree".format(self.config.save_dir, self.config.dataset)
        self.cls_reps = np.load(os.path.join(index_dir, "cls_reps.npy"))
        self.token_reps = np.load(os.path.join(index_dir, "token_reps.npy"))
        self.token_d_ids = np.load(os.path.join(index_dir, "token_d_ids.npy"))
        self.init_isax_mtree()

    def init_isax_mtree(self):
        config = iSaxMtree.Config()
        config.save_dir = r"{}/{}/isax_mtree".format(self.config.save_dir, self.config.dataset)
        config.num_data =  self.token_reps.shape[0]
        config.highDim =  self.token_reps.shape[1] + 1
        config.lowDim  = self.config.low_dim
        config.search_Radius = self.config.search_Radius
        config.KNN = self.config.KNN
        config.c_appro = self.config.c_appro
        config.T = config.num_data * 0.1
        config.alpha1 = self.config.alpha1
        config.t = iSaxMtree.MyFunc.Ccal_thresh(config.lowDim, config.alpha1)
        self.vector_db = iSaxMtree.VectorDB(config)
        self.vector_db.initVectorDB()

    def mtree_search(self, q_reprs):
        all_r_repr, all_r_lens, all_q_lens, all_r_ids = [], [], [], []
        for q_v in q_reprs:
            q_v = q_v.detach().numpy().reshape(1, -1)
            r_vectors, r_d_ids = self.get_related_d_v(q_v)
            all_r_repr.append(r_vectors)
            all_r_lens.append(len(r_vectors))
            all_q_lens.append(1)
            all_r_ids.append(torch.tensor(r_d_ids).to(torch.int64))
        if len(all_r_repr) > 0:
            all_r_repr = torch.tensor(np.concatenate(all_r_repr, axis=0)).to(torch.float32)
        return all_r_repr, all_r_lens, all_q_lens, all_r_ids

    def search(self, q_reprs):
        self.reset_sum_scores()
        cls_rep = q_reprs[0][0]
        self.cls_search(cls_rep)
        self.token_search(q_reprs[0,1:])
        top_scores, top_ids = self.select_top_k(self.topk)
        self.reset_sum_scores()
        return top_scores, top_ids

    def cls_search(self, cls_rep):
        d_cls_reps = torch.tensor(self.cls_reps).to(torch.float32).to("cpu")
        self.sum_scores = torch.matmul(cls_rep, d_cls_reps.T).unsqueeze(0)


    def get_related_d_v(self, query_hash_value):
        query_v = np.append(query_hash_value[0], 0)
        vector_id =  self.vector_db.search(query_v).tolist()
        return self.token_reps[vector_id], self.token_d_ids[vector_id]

    def token_search(self, q_reprs):
        all_r_reprs, all_r_lens, all_q_lens, all_r_ids = self.mtree_search(q_reprs)
        if len(all_r_reprs) > 0:
            # compute similarity
            try:
                all_batch_scores = IsaxMTreeSearcher.compute_similarity(q_reprs, all_r_reprs)
            except Exception as e:
                print(e)
            q_start, ctx_start = 0, 0
            for i, (q_len, ctx_len) in enumerate(zip(all_q_lens, all_r_lens)):
                q_end, ctx_end = q_start + q_len, ctx_start + ctx_len
                scores = all_batch_scores[q_start:q_end, ctx_start:ctx_end][0]
                ctx_id = all_r_ids[i]
                torch_scatter.scatter_max(src=scores, index=ctx_id, out=self.max_scores, dim=-1)
                self.sum_scores += self.max_scores
                self.max_scores.fill_(0)
                q_start = q_end
                ctx_start = ctx_end

    def select_top_k(self, top_k):
        top_scores, top_ids = self.sum_scores.topk(top_k, dim=1)
        return top_scores, top_ids

    def reset_sum_scores(self):
        self.sum_scores.fill_(0)


    @staticmethod
    def compute_similarity(q_repr, ctx_repr):
        return torch.matmul(q_repr, ctx_repr.T).relu()