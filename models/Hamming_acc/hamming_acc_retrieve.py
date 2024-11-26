import gzip
import json
import os
import perf_event

import faiss
import ir_measures
import pandas as pd
import torch
from tqdm import tqdm

from models.Hamming.hamming_retrieve import HammingRetrieve
from models.Hamming_acc.hamming_acc_search import HammingAccSearcher
from utils.retrieve_utils import create_this_perf

columns = ["encode_cycles", "encode_instructions",
           "encode_L1_misses", "encode_LLC_misses",
           "encode_L1_accesses", "encode_LLC_accesses",
           "encode_branch_misses", "encode_task_clock",
           "retrieval_cycles", "retrieval_instructions",
           "retrieval_L1_misses", "retrieval_LLC_misses",
           "retrieval_L1_accesses", "retrieval_LLC_accesses",
           "retrieval_branch_misses", "retrieval_task_clock"]


class HammingAccRetrieve(HammingRetrieve):
    def __init__(self, config):
        super().__init__(config)
        self.perf_df = None
        self.eval_results = None

    def prepare_searcher(self):
        self.searcher = HammingAccSearcher(self.config, len(self.corpus.corpus_list))
        self.searcher.prepare_index()

    def retrieve(self):
        self._create_save_path()
        all_query_match_scores = []
        all_query_inids = []
        all_perf = []
        for query in tqdm(list(self.queries.values())):
            perf_encode = perf_event.PerfEvent()
            perf_retrival = perf_event.PerfEvent()
            query = [query]
            perf_encode.startCounters()
            Q_reps = self.context_encoder.queryFromText(query, bsize=None, to_cpu=True,
                                                           full_length_search=False)
            perf_encode.stopCounters()

            perf_retrival.startCounters()
            top_scores, top_ids = self.searcher.search(Q_reps)
            perf_retrival.stopCounters()

            this_perf = create_this_perf(perf_encode, perf_retrival)
            all_perf.append(this_perf)
            all_query_match_scores.append(top_scores)
            all_query_inids.append(top_ids)
        all_query_match_scores = torch.cat(all_query_match_scores, dim=0)
        all_query_exids = torch.cat(all_query_inids, dim=0)
        self.perf_df = pd.DataFrame(all_perf, columns=columns)
        path = self.save_ranks(all_query_match_scores, all_query_exids)
        return path

    def evaluation(self, path):
        new_2_old = list(self.corpus.corpus.keys())
        rank_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))
        for i, r in rank_results_pd.iterrows():
            rank_results_pd.at[i, "doc_id"] = new_2_old[int(r["doc_id"])]
        self.eval_results = ir_measures.calc_aggregate(self.config.measure, self.labels, rank_results_pd)

    def save_ranks(self, scores, indices):
        path = r"{}/hamming_acc_{}.run.gz".format(self.rank_path, self.config.hash_dimmension)
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

    def save_perf(self):
        self.perf_df.to_csv(r"{}/perf_{}.csv".format(self.perf_path, self.config.hash_dimmension),index=False)

    def _create_save_path(self):
        save_dir = r"{}/hamming_acc/{}".format(self.config.results_save_to, self.config.dataset)
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

    def save_metadata(self):
        num_tokens = 0
        for bin in self.searcher.hash_bins.values():
            num_tokens += int(bin['dense_repr'][0].shape[0])

        metadata = {"num_docs": len(list(self.corpus.corpus.keys())),
                    "num_tokens": num_tokens}
        save_dir = r"{}/hamming_acc/{}".format(self.config.results_save_to, self.config.dataset)
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)


    def run(self):
        self.setup()
        path = self.retrieve()
        self.evaluation(path)
        self.save_perf()
        self.save_metadata()


