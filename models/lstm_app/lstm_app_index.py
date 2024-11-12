import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from dataloader.colbert_dataloader import ColbertDataset
from encoder.colbert_encoder import ColbertEncoder
from models.base_model import BaseIndex
from models.lstm_app.lsh_database import LSHDatabase
from models.lstm_app.tree_index import LSHTreeIndex
from utils.colbert_utils import batch


class LSTMAPPIndex(BaseIndex):
    def __init__(self, config):
        super().__init__(config)
        self.lsh_database = None
        self.tree_index = None

    def prepare_data(self):
        corpus_path = r"{}/corpus/{}.jsonl".format(self.config.root_dir, self.config.dataset)
        self.dataset = ColbertDataset(corpus_path)

    def prepare_model(self):
        self.context_encoder = ColbertEncoder(self.config)

    def init_lsh_database(self):
        print("Initializing LSH database")
        self.lsh_database = LSHDatabase(self.config, self.config.dim)
        self.lsh_database.create_random_hash_matrix()
        print("Finished initializing LSH database")

    def init_tree_index(self):
        assert self.lsh_database is not None
        self.tree_index = LSHTreeIndex(self.config, self.lsh_database)
        self.tree_index.build_tree()

    def setup(self):
        self.prepare_data()
        self.prepare_model()
        self.init_lsh_database()
        self.init_tree_index()


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
        assert self.tree_index is not None
        assert self.lsh_database is not None
        assert self.embs_list is not None
        self.all_embs = torch.cat(self.embs_list, dim=0)
        self.token_labels = []
        self.cls_reps = []
        self.token_reps = []
        offsets = [0]
        for doclen in self.doclens_list:
            offsets.append(offsets[-1] + doclen)

        i = 0
        n = 0
        d_id = 0
        for token_id, embs in enumerate(tqdm(self.all_embs)):
            if token_id in offsets:
                self.cls_reps.append(embs.unsqueeze(0))
            else:
                token_repr = embs.unsqueeze(0).detach().numpy()
                self.lsh_database.token_reps.append(token_repr)
                self.lsh_database.token_d_ids.append(d_id)
                self.tree_index.insert(token_repr, i)
                n +=1
                i +=1
                if n == self.doclens_list[d_id] - 1:
                    n = 0
                    d_id += 1

        self.lsh_database.token_reps = np.concatenate(self.lsh_database.token_reps, axis=0)
        self.lsh_database.token_d_ids = np.array(self.lsh_database.token_d_ids)


    def save_index(self):
        save_path = r"{}/index/{}/lstm_app".format(self.config.save_dir, self.config.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(self.lsh_database.token_reps, os.path.join(save_path, "token_reps.npy"))
        np.save(self.lsh_database.token_d_ids, os.path.join(save_path, "token_d_ids.npy"))
        np.save(self.lsh_database.hash_matrix, os.path.join(save_path, "hash_matrix.npy"))

        with open(os.path.join(save_path, "tree_index.pkl"), "wb") as f:
            pickle.dump(self.tree_index, f)
