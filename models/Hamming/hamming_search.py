import os
import torch
import torch_scatter
from tqdm import tqdm

from models.base_model import BaseSearcher


class HammingSearcher(BaseSearcher):
    def __init__(self, config, num_doc):
        super().__init__(config)
        self.hamming_threshold = config.hamming_threshold
        self.cls_reps = None
        self.token_labels = None
        self.hash_matrix = None
        self.hash_bins = None
        self.num_doc = num_doc
        self.topk = config.topk
        self.sum_scores = torch.zeros((1, self.num_doc,), dtype=torch.float32)
        self.max_scores = torch.zeros((self.num_doc,), dtype=torch.float32)

    def prepare_index(self):
        index_dir = r"{}/index/{}/hamming".format(self.config.save_dir, self.config.dataset)
        self.cls_reps = torch.load(r"{}/{}".format(index_dir, 'cls_reps.pt'), map_location="cpu")
        self.token_labels = torch.load(r"{}/{}".format(index_dir, 'token_labels.pt'), map_location="cpu")
        self.hash_matrix = torch.load(r"{}/{}".format(index_dir, 'hash_matrix.pt'), map_location="cpu")
        self.hash_bins = torch.load(r"{}/{}".format(index_dir, 'hash_bins.pt'), map_location="cpu")

    def search(self, q_reprs):
        self.reset_sum_scores()
        cls_rep = q_reprs[0][0]
        self.cls_search(cls_rep)
        self.lsh_search(q_reprs[:,1:])
        top_scores, top_ids = self.select_top_k()
        self.reset_sum_scores()
        return top_scores, top_ids

    def cls_search(self, cls_rep):
        self.sum_scores = torch.matmul(cls_rep, self.cls_reps.T).unsqueeze(0)



    def lsh_search(self, embeddings):
        hamming_matrix = self.cal_hash_value(embeddings).squeeze()
        for i, hamming_key in enumerate(tqdm(hamming_matrix)):
            hash_value = "".join(hamming_key.numpy().astype(str))
            similar_hash_bin = self.find_similar_keys(hash_value)
            if len(similar_hash_bin.keys()) == 0:
                continue
            doc_token_reps = []
            token_pid = []
            for value in similar_hash_bin.values():
                doc_token_reps.append(value["dense_repr"][0])
                token_pid += value["dense_repr"][1]

            doc_token_reps = torch.cat(doc_token_reps, dim=0)
            token_score = torch.matmul(embeddings[0][i], doc_token_reps.T).relu_()
            token_pid_tensor = torch.Tensor(token_pid).to(torch.int64)
            torch_scatter.scatter_max(src=token_score, index=token_pid_tensor, out=self.max_scores, dim=-1)
            self.sum_scores += self.max_scores
            self.max_scores.fill_(0)

    def select_top_k(self):
        top_scores, top_ids = self.sum_scores.topk(self.topk, dim=1)
        return top_scores, top_ids

    def reset_sum_scores(self):
        self.sum_scores.fill_(0)

    def find_similar_keys(self, query):
        similar_keys_values = {}
        for key, value in self.hash_bins.items():
            if HammingSearcher.hamming_distance(query, key) <= self.hamming_threshold:
                similar_keys_values[key] = value
        return similar_keys_values

    def cal_hash_value(self, embeddings):
        hash_value_matrix = embeddings @ self.hash_matrix
        hamming_matrix = torch.where(hash_value_matrix > 0, torch.tensor(1), torch.tensor(0))
        return hamming_matrix

    @staticmethod
    def hamming_distance(s1,s2):
        if len(s1) != len(s2):
            raise ValueError("Length of s1 must be equal to length of s2.")
        return sum(c1!=c2 for c1,c2 in zip(s1,s2))





