import gzip
import json
import os
from abc import ABC

import faiss
import ir_measures
import torch
import pandas as pd
from tqdm import tqdm

from dataloader.colbert_dataloader import ColbertDataset
from encoder.colbert_encoder import ColbertEncoder
from models.Hamming.hamming_search import HammingSearcher
from models.all.all_search import AllSearcher
from models.base_model import BaseRetrieve

class HammingRetrieve(BaseRetrieve):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        self.prepare_model()
        self.prepare_data()
        self.prepare_searcher()


    def prepare_data(self):
        with open(r"{}/query/{}.json".format(self.config.root_dir, self.config.dataset), 'r', encoding="utf-8") as f:
            self.queries = json.load(f)
        self.labels = pd.read_csv(r"{}/label/{}.csv".format(self.config.root_dir, self.config.dataset))
        self.labels["query_id"] = self.labels["query_id"].astype(str)
        self.labels["doc_id"] = self.labels["doc_id"].astype(str)
        corpus_path = r"{}/corpus/{}.jsonl".format(self.config.root_dir, self.config.dataset)
        self.corpus = ColbertDataset(corpus_path)

    def prepare_model(self):
        self.context_encoder = ColbertEncoder(self.config)

    def prepare_searcher(self):
        self.searcher = HammingSearcher(self.config, len(self.corpus.corpus_list))
        self.searcher.prepare_index()

    def retrieve(self):
        self._create_save_path()
        all_query_match_scores = []
        all_query_inids = []
        for query in tqdm(list(self.queries.values())):
            query = [query]
            Q_reps = self.context_encoder.queryFromText(query, bsize=None, to_cpu=True,
                                                           full_length_search=False)
            top_scores, top_ids = self.searcher.search(Q_reps)
            all_query_match_scores.append(top_scores)
            all_query_inids.append(top_ids)
        all_query_match_scores = torch.cat(all_query_match_scores, dim=0)
        all_query_exids = torch.cat(all_query_inids, dim=0)
        path = self.save_ranks(all_query_match_scores, all_query_exids)
        return path

    def evaluation(self, path):
        new_2_old = list(self.corpus.corpus.keys())
        rank_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))
        for i, r in rank_results_pd.iterrows():
            rank_results_pd.at[i, "doc_id"] = new_2_old[int(r["doc_id"])]
        eval_results = ir_measures.calc_aggregate(self.config.measure, self.labels, rank_results_pd)
        return eval_results

    def save_ranks(self, scores, indices):
        path = r"{}/all.run.gz".format(self.rank_path)
        rh = faiss.ResultHeap(scores.shape[0], self.config.topk)

        rh.add_result(-scores.numpy(), indices.numpy())

        rh.finalize()
        corpus_scores, corpus_indices = (-rh.D).tolist(), rh.I.tolist()

        qid_list = list(self.queries.keys())
        with gzip.open(path, 'wt') as fout:
            for i in range(len(corpus_scores)):
                q_id = qid_list[i]
                scores = corpus_scores[i]
                indices = corpus_indices[i]
                for j in range(len(scores)):
                    fout.write(f'{q_id} 0 {indices[j]} {j} {scores[j]} run\n')
        return path

    def _create_save_path(self):
        save_dir = r"{}/hamming/{}".format(self.config.results_save_to, self.config.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.perf_path = r"{}/{}".format(save_dir, "perf_results")
        self.rank_path = r"{}/{}".format(save_dir, "rank_results")
        self.eval_path = r"{}/{}".format(save_dir, "eval_results")
        if not os.path.exists(self.perf_path):
            os.makedirs(self.perf_path)
        if not os.path.exists(self.rank_path):
            os.makedirs(self.rank_path)
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)