import os

from tqdm import tqdm
import torch

from dataloader.colbert_dataloader import ColbertDataset
from encoder.colbert_encoder import ColbertEncoder
from models.base_model import BaseIndex
from utils.colbert_utils import batch


class HammingIndex(BaseIndex):
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
        self.cls_reps = []
        self.token_reps = []
        offsets = [0]
        for doclen in self.doclens_list:
            offsets.append(offsets[-1] + doclen)

        for token_id, embs in enumerate(tqdm(self.all_embs)):
            if token_id in offsets:
                self.cls_reps.append(embs.unsqueeze(0))
            else:
                self.token_reps.append(embs.unsqueeze(0))

        for id, num in enumerate(self.doclens_list):
            for _ in range(num-1):
                self.token_labels.append(id)

        self.cls_reps = torch.cat(self.cls_reps, dim=0)
        self.token_reps = torch.cat(self.token_reps, dim=0)
        
        self.hamming_hashing()

    def hamming_hashing(self):
        self.hash_matrix = torch.rand(self.token_reps.shape[1], self.config.hash_dimmension)
        hash_value_matrix = self.token_reps @ self.hash_matrix
        hamming_matrix = torch.where(hash_value_matrix > 0, torch.tensor(1), torch.tensor(0))
        self.hash_bins = {}
        for i, row in enumerate(tqdm(hamming_matrix)):
            hash_value = "".join(row.numpy().astype(str))
            if hash_value in list(self.hash_bins.keys()):
                self.hash_bins[hash_value]["dense_repr"].append((self.token_reps[i], self.token_labels[i]))
            else:
                self.hash_bins[hash_value] = {}
                self.hash_bins[hash_value]["dense_repr"] = [(self.token_reps[i], self.token_labels[i])]
        for hash_value, values in self.hash_bins.items():
            self.hash_bins[hash_value]["dense_repr"] = Matrixing(values["dense_repr"])

    def save_index(self):
        save_path = r"{}/index/{}/hamming".format(self.config.save_dir, self.config.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.hash_matrix, os.path.join(save_path, "hash_matrix.pt"))
        torch.save(self.hash_bins, os.path.join(save_path, "hash_bins.pt"))
        torch.save(self.token_labels, os.path.join(save_path, "token_labels.pt"))
        torch.save(self.cls_reps, os.path.join(save_path, "cls_reps.pt"))

    def setup(self):
        self.prepare_model()
        self.prepare_data()

    def fit(self):
        self.encode()
        self.indexing()
        self.save_index()

def Matrixing(repr_pair_list):
    repr = torch.stack([repr_pair[0] for repr_pair in repr_pair_list], dim=0)
    ivd = [repr_pair[1] for repr_pair in repr_pair_list]
    return (repr,ivd)



