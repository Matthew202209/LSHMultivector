import os

from models.Hamming.hamming_index import HammingIndex, Matrixing
from tqdm import tqdm
import torch


class HammingAccIndex(HammingIndex):
    def __init__(self, config):
        super().__init__(config)

    def hamming_hashing(self):
        self.hash_matrix = torch.rand(self.token_reps.shape[1], self.config.hash_dimmension)

        hash_value_matrix = self.token_reps @ self.hash_matrix
        hamming_matrix = torch.where(hash_value_matrix > 0, torch.tensor(1), torch.tensor(0))
        self.hash_bins = {format(key, '0{}b'.format(self.config.hash_dimmension)): {"dense_repr": []} for key in range(2**self.config.hash_dimmension)}

        for i, row in enumerate(tqdm(hamming_matrix)):
            hash_value = "".join(row.numpy().astype(str))
            self.hash_bins[hash_value]["dense_repr"].append((self.token_reps[i], self.token_labels[i]))
        for hash_value in list(self.hash_bins.keys()):
            if len(self.hash_bins[hash_value]["dense_repr"]) == 0:
                del self.hash_bins[hash_value]
                continue
            self.hash_bins[hash_value]["dense_repr"] = Matrixing(self.hash_bins[hash_value]["dense_repr"])

    def save_index(self):
        save_path = r"{}/index/{}/hamming_acc/{}".format(self.config.save_dir, self.config.dataset, self.config.hash_dimmension)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.hash_matrix, os.path.join(save_path, "hash_matrix.pt"))
        torch.save(self.hash_bins, os.path.join(save_path, "hash_bins.pt"))
        torch.save(self.token_labels, os.path.join(save_path, "token_labels.pt"))
        torch.save(self.cls_reps, os.path.join(save_path, "cls_reps.pt"))


