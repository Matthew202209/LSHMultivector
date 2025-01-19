import os
import torch
import torch_scatter

from models.base_model import BaseSearcher


class AllClsSearcher(BaseSearcher):
    def __init__(self, config, num_doc):
        super().__init__(config)
        self.cls_reps = None
        self.token_reps = None
        self.token_label = None

        self.num_doc = num_doc
        self.topk = config.topk
        self.sum_scores = torch.zeros((1, self.num_doc,), dtype=torch.float32)
        self.max_scores = torch.zeros((self.num_doc,), dtype=torch.float32)

    def prepare_index(self):
        index_dir = r"{}/index/{}/all_cls".format(self.config.save_dir, self.config.dataset)
        self.cls_reps = torch.load(r"{}/{}".format(index_dir,  "all_cls_cls_reps.pt"), map_location="cpu")
        self.token_reps = torch.load(r"{}/{}".format(index_dir, "all_cls_token_reps.pt"), map_location="cpu")
        self.token_label = torch.load(r"{}/{}".format(index_dir, "token_label.pt"), map_location="cpu")

    # def search(self, q_reprs):
    #     self.reset_sum_scores()
    #     for q_repr in q_reprs[0]:
    #         token_score = torch.matmul(q_repr, self.all_embs.T).relu_()
    #         torch_scatter.scatter_max(src=token_score, index= self.token_label, out=self.max_scores, dim=-1)
    #         self.sum_scores[0] += self.max_scores
    #         self.max_scores.fill_(0)
    #     top_scores, top_ids = self.select_top_k()
    #     self.reset_sum_scores()
    #     return top_scores, top_ids

    def cls_search(self, cls_rep):
        d_cls_reps = torch.tensor(self.cls_reps).to(torch.float32).to("cpu")
        self.sum_scores = torch.matmul(cls_rep, d_cls_reps.T).unsqueeze(0)

    def search(self, q_reprs):
        self.reset_sum_scores()
        cls_rep = q_reprs[0][0]
        self.cls_search(cls_rep)
        self.token_search(q_reprs[0,1:])
        top_scores, top_ids = self.select_top_k(self.topk)
        self.reset_sum_scores()
        return top_scores, top_ids

    def token_search(self, q_reprs):
        for q_repr in q_reprs:
            token_score = torch.matmul(q_repr, self.token_reps.T).relu_()
            torch_scatter.scatter_max(src=token_score, index= self.token_label, out=self.max_scores, dim=-1)
            self.sum_scores[0] += self.max_scores
            self.max_scores.fill_(0)

    def select_top_k(self, topk):
        top_scores, top_ids = self.sum_scores.topk(topk, dim=1)
        return top_scores, top_ids

    def reset_sum_scores(self):
        self.sum_scores.fill_(0)





