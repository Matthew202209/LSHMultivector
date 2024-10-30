import itertools
import os

from models.Hamming.hamming_index import HammingIndex, Matrixing
from tqdm import tqdm
import torch

from utils.hamming_utils import generate_hamming_codes
from utils.trie_tree_utils import Trie


class HammingAccIndex(HammingIndex):
    def __init__(self, config):
        super().__init__(config)
        self.hash_matrix = None
        self.hash_ivd = {}
        self.hash_bins = {}

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
            for _ in range(num - 1):
                self.token_labels.append(id)

        self.cls_reps = torch.cat(self.cls_reps, dim=0)
        self.token_reps = torch.cat(self.token_reps, dim=0)

    def reset(self):
        self.hash_bins = {}
        self.hash_ivd = {}
        self.hash_matrix = None

    def fit(self):
        self.hamming_hashing()
        self.save_index()
        self.reset()


    def hamming_hashing(self):
        self.hash_matrix = torch.rand(self.token_reps.shape[1], self.config.hash_dimmension)

        hash_value_matrix = self.token_reps @ self.hash_matrix
        hamming_matrix = torch.where(hash_value_matrix > 0, torch.tensor(1), torch.tensor(0))
        self.hash_bins = {format(key, '0{}b'.format(self.config.hash_dimmension)): {"dense_repr": []} for key in range(2**self.config.hash_dimmension)}

        for i, row in enumerate(tqdm(hamming_matrix)):
            hash_value = "".join(row.numpy().astype(str))
            self.hash_bins[hash_value]["dense_repr"].append((self.token_reps[i], self.token_labels[i]))
        for hash_value in tqdm(list(self.hash_bins.keys())):

            if len(self.hash_bins[hash_value]["dense_repr"]) == 0:
                del self.hash_bins[hash_value]
                continue
            self.hash_bins[hash_value]["dense_repr"] = Matrixing(self.hash_bins[hash_value]["dense_repr"])
            # self.hash_ivd[hash_value] = generate_hamming_codes(hash_value, self.config.hamming_threshold)
        for hash_value in generate_binary_numbers(self.config.hash_dimmension):
            self.hash_ivd[hash_value] = generate_hamming_codes(hash_value, self.config.hamming_threshold)





    def save_index(self):
        save_path = r"{}/index/{}/hamming_acc/{}".format(self.config.save_dir, self.config.dataset, self.config.hash_dimmension)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.hash_matrix, os.path.join(save_path, "hash_matrix.pt"))
        torch.save(self.hash_bins, os.path.join(save_path, "hash_bins.pt"))
        torch.save(self.token_labels, os.path.join(save_path, "token_labels.pt"))
        torch.save(self.cls_reps, os.path.join(save_path, "cls_reps.pt"))
        torch.save(self.hash_ivd, os.path.join(save_path, "hash_ivd.pt"))



def generate_binary_numbers(n):
    return ["".join(bits) for bits in itertools.product("01", repeat=n)]