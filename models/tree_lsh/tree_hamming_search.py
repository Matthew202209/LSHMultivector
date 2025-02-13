import os

import dill as pickle
import torch
import torch_scatter

from models.base_model import BaseSearcher
from models.lstm_app.lsh_database import LSHDatabase
from models.tree_lsh.tree_hamming_lsh_database import TreeHammingDatabase


class TreeHammingSearcher(BaseSearcher):
    def __init__(self, config, num_doc):
        super().__init__(config)
        self.tree_index = None
        self.lsh_database = None
        self.num_doc = num_doc
        self.topk = config.topk
        self.sum_scores = torch.zeros((1, self.num_doc,), dtype=torch.float32)
        self.max_scores = torch.zeros((self.num_doc,), dtype=torch.float32)

    def prepare_index(self):
        if self.config.version == "v1":
            index_dir = r"{}/index/{}/tree_hamming_v1".format(self.config.save_dir, self.config.dataset)
        elif self.config.version == "v2":
            index_dir = r"{}/index/{}/tree_hamming_v2".format(self.config.save_dir, self.config.dataset)
        self.load_tree_index(index_dir)
        self.load_lsh_database(index_dir)

    def load_tree_index(self, index_dir):
        tree_index_path = os.path.join(index_dir, 'tree_index.pkl')
        with open(tree_index_path, 'rb') as f:
            self.tree_index = pickle.load(f)

    def load_lsh_database(self, index_dir):
        self.lsh_database = TreeHammingDatabase(self.config, self.config.dim)
        self.lsh_database.load_index(index_dir)


    def search(self, q_reprs):
        self.reset_sum_scores()
        cls_rep = q_reprs[0][0]
        self.cls_search(cls_rep)
        _, coarse_top_ids = self.select_top_k(self.config.coarse_top_k)
        indices_to_keep = coarse_top_ids[0].detach().cpu().numpy().tolist()
        # mask = torch.zeros_like(self.sum_scores, dtype=torch.bool)
        # mask[:, indices_to_keep] = True
        self.lsh_search(q_reprs[0,1:], indices_to_keep)
        # self.sum_scores = self.sum_scores * mask
        top_scores, top_ids = self.select_top_k(self.topk)
        self.reset_sum_scores()
        return top_scores, top_ids


    def cls_search(self, cls_rep):
        d_cls_reps = torch.tensor(self.lsh_database.cls_reps).to(torch.float32).to("cpu")
        self.sum_scores = torch.matmul(cls_rep, d_cls_reps.T).unsqueeze(0)

    def lsh_search(self, q_reprs, indices_to_keep):
        all_r_reprs, all_r_lens, all_q_lens, all_r_ids = self.tree_index.hash_search(q_reprs, indices_to_keep)
        if len(all_r_reprs) > 0:
            # compute similarity
            try:
                all_batch_scores = TreeHammingSearcher.compute_similarity(q_reprs, all_r_reprs)
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