import time
from distutils.command.build import build

import numpy as np
import torch
from tqdm import tqdm

from models.tree_lsh.tree_hamming_lsh_database import TreeHammingDatabase


class TreeNote:
    def __init__(self, layer, id):
        self.is_leaf = False
        self.layer = layer
        self.id = id
        self.children = []
        self.binary_index = [{},{},{}]
        self.vectors_indexes = []
        self.document_id = set()

    def check_document_exist(self, reference_set):
        return not self.document_id.isdisjoint(reference_set)

class TreeHammingTreeIndex:
    def __init__(self, config, lsh_database:TreeHammingDatabase):
        self.root = TreeNote(0,0)
        self.lsh_database = lsh_database
        self.tree_layers = config.tree_layers
        self.hash_dim = config.hash_dim
        self.hash_values = None
        self.related_d_v_index = []

    def build_tree(self):
        self.build_next_layer(self.root, 1, 0)

    def build_next_layer(self, note, layer, father_id):
        for id in range(2**self.hash_dim):
            this_id = father_id *2**((layer-1)*self.hash_dim) + id
            new_note = TreeNote(layer, this_id)
            if layer != self.tree_layers-1:
                self.build_next_layer(new_note, layer+1, this_id)
                note.children.append(new_note)
            else:
                new_note.is_leaf = True
                note.children.append(new_note)

    def insert(self, vector, vector_index, document_id):
        self.cal_hash_values(vector)
        self.set_binary_index(self.root, vector_index, document_id = document_id)

    def set_binary_index(self, note, vector_index, document_id = None):
        note.document_id.add(document_id)
        if note.is_leaf:
            note.vectors_indexes.append(vector_index)
        else:
            new_hash_value = self.get_new_hash_value(note.layer, note.id, self.hash_dim)
            segment1 = TreeHammingTreeIndex.split_binary_string(new_hash_value,0,4)
            segment2 = TreeHammingTreeIndex.split_binary_string(new_hash_value,2,6)
            segment3 = TreeHammingTreeIndex.get_binary_string(new_hash_value, 0, 1, 4, 5)
            TreeHammingTreeIndex.init_binary_index(note, segment1, segment2, segment3)
            # TreeHammingTreeIndex.init_binary_index(note, segment1, segment2)
            note.binary_index[0][segment1].add(note.children[int(new_hash_value, 2)])
            note.binary_index[1][segment2].add(note.children[int(new_hash_value, 2)])
            note.binary_index[2][segment3].add(note.children[int(new_hash_value, 2)])


            self.set_binary_index(note.children[int(new_hash_value, 2)], vector_index, document_id = document_id)

    def hash_search(self, all_q_v, candidate_d_list):
        all_r_repr, all_r_lens, all_q_lens, all_r_ids= [], [], [], []
        for q_v in all_q_v:
            q_v = q_v.detach().numpy().reshape(1,-1)
            r_vectors, r_d_ids = self.get_related_d_v(q_v, candidate_d_list)
            self.reset_related_d_v_index()
            all_r_repr.append(r_vectors)
            all_r_lens.append(len(r_vectors))
            all_q_lens.append(1)
            all_r_ids.append(torch.tensor(r_d_ids).to(torch.int64))
        if len(all_r_repr) > 0:
            all_r_repr = torch.tensor(np.concatenate(all_r_repr, axis = 0)).to(torch.float32)
        return all_r_repr, all_r_lens, all_q_lens, all_r_ids


    def get_related_d_v(self, q_v, candidate_d_list):
        assert self.hash_values is not None
        self.cal_hash_values(q_v)
        # d_vectors = self.lsh_database.token_reps
        # token_d_ids = self.lsh_database.token_d_ids
        self.get_d_v_index(self.root, candidate_d_list)
        assert len(self.related_d_v_index) != 0
        related_d_v_index = list(set(self.related_d_v_index))
        return self.lsh_database.token_reps[related_d_v_index], self.lsh_database.token_d_ids[related_d_v_index]

    def get_d_v_index(self, note, candidate_d_list):
        children = []
        if note.is_leaf:
            self.related_d_v_index += note.vectors_indexes
        else:
            new_hash_value = self.get_new_hash_value(note.layer, note.id, self.hash_dim)
            segment1 = TreeHammingTreeIndex.split_binary_string(new_hash_value, 0, 4)
            segment2 = TreeHammingTreeIndex.split_binary_string(new_hash_value, 2, 6)
            segment3 = TreeHammingTreeIndex.get_binary_string(new_hash_value, 0, 1, 4,5)
            try:
                children += list(note.binary_index[0][segment1])
            except:
                pass
            try:
                children += list(note.binary_index[1][segment2])
            except:
                pass
            try:
                children += list(note.binary_index[2][segment3])
            except:
                pass
            # for child in children:
            #     if child.check_document_exist(candidate_d_list):
            #         self.get_d_v_index(child, candidate_d_list)
            #     else:
            #         continue
            for child in children:
                self.get_d_v_index(child, candidate_d_list)


    def cal_hash_values(self, vector):
        self.hash_values  = np.tensordot(vector , self.lsh_database.hash_matrix, axes=([1],[2]))[0]
        self.hash_values = (self.hash_values > 0).astype(int)

    def get_new_hash_value(self, layer, id, hash_dim):
        return "".join(str(bit) for bit in self.hash_values[layer, id*hash_dim:id * hash_dim+hash_dim])

    def reset_related_d_v_index(self):
        self.related_d_v_index = []

    @staticmethod
    def split_binary_string(binary_string, start_index, end_index):
        return binary_string[start_index:end_index]

    @staticmethod
    def get_binary_string(binary_string, i1,i2,i3,i4):
        new_binary_string = "".join([str(binary_string[i1]), str(binary_string[i2]),
                                     str(binary_string[i3]), str(binary_string[i4])])


        return new_binary_string


    @staticmethod
    def init_binary_index(note, segment1, segment2, segment3):
        if segment1 not in list(note.binary_index[0].keys()):
            note.binary_index[0][segment1] = set()
        if segment2 not in list(note.binary_index[1].keys()):
            note.binary_index[1][segment2] = set()
        if segment3 not in list(note.binary_index[2].keys()):
            note.binary_index[2][segment3] = set()



