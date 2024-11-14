import time
from distutils.command.build import build

import numpy as np
import torch

from models.lstm_app.lsh_database import LSHDatabase

class TreeNote:
    def __init__(self, layer, id):
        self.is_leaf = False
        self.layer = layer
        self.id = id
        self.children = []
        self.binary_index = [{},{},{}]
        self.vectors_indexes = []

class LSHTreeIndex:
    def __init__(self, config, lsh_database:LSHDatabase):
        self.root = TreeNote(0,0)
        self.lsh_database = lsh_database
        self.tree_layers = config.tree_layers
        self.first_layer_hash_dim = config.first_layer_hash_dim
        self.hash_dim = config.hash_dim
        self.hash_values = None
        self.related_d_v_index = []

    def build_tree(self):
    # first layer
        for id in range(2**self.first_layer_hash_dim):
            first_layer_note = TreeNote(2, id)
            self.build_other_layer(first_layer_note, 3, id)
            self.root.children.append(first_layer_note)

    def build_other_layer(self, note, layer, father_id):
        for id in range(2**self.hash_dim):
            this_id = id+father_id*2**self.hash_dim
            new_note = TreeNote(layer, this_id)
            if layer != self.tree_layers:
                self.build_other_layer(new_note, layer+1, this_id)
                note.children.append(new_note)
            else:
                new_note.is_leaf = True
                note.children.append(new_note)

    def insert(self, vector, vector_index):
        self.cal_hash_values(vector)
        hash_value = self.get_new_hash_value(1,0, self.first_layer_hash_dim)
        self.set_binary_index(self.root, vector_index, hash_value = hash_value)

    def set_binary_index(self, note, vector_index, hash_value = None):
    # first layer
        if note.layer == 0:
            note.binary_index[0][hash_value] = note.children[int(hash_value,2)]
            for child in note.children:
                self.set_binary_index(child, vector_index)
        else:
            #todo setting as parameterization

            if note.is_leaf:
                note.vectors_indexes.append(vector_index)
            else:
                new_hash_value = self.get_new_hash_value(note.layer, note.id, self.hash_dim)
                segment1 = LSHTreeIndex.split_binary_string(new_hash_value,0,4)
                segment2 = LSHTreeIndex.split_binary_string(new_hash_value,2,6)
                segment3 = LSHTreeIndex.split_binary_string(new_hash_value,4,8)
                LSHTreeIndex.init_binary_index(note, segment1, segment2, segment3)

                note.binary_index[0][segment1].add(note.children[int(new_hash_value, 2)])
                note.binary_index[1][segment2].add(note.children[int(new_hash_value, 2)])
                note.binary_index[2][segment3].add(note.children[int(new_hash_value, 2)])
                self.set_binary_index(note.children[int(new_hash_value, 2)], vector_index)

    def hash_search(self, all_q_v):
        all_r_repr, all_r_lens, all_r_ids= [], [], []
        for q_v in all_q_v:
            q_v = q_v.detach().numpy()
            r_vectors, r_d_ids = self.get_related_d_v(q_v)
            all_r_repr.append(r_vectors)
            all_r_lens.append(len(r_vectors))
            all_r_ids.append(r_d_ids)
        if len(all_r_repr) > 0:
            all_r_repr = torch.tensor(np.concatenate(all_r_repr, axis = 0)).to(torch.float32)
        return all_r_repr, all_r_lens, all_r_ids


    def get_related_d_v(self, q_v):
        assert self.hash_values is None
        self.cal_hash_values(q_v)
        d_vectors = self.lsh_database.token_reps
        token_d_ids = self.lsh_database.token_d_ids
        # first layer
        first_hash_value = "".join(str(bit) for bit in self.hash_values[1,0,0:self.first_layer_hash_dim])
        child_note = self.root.binary_index[0][first_hash_value]
        self.get_d_v_index(child_note)
        assert len(self.related_d_v_index) != 0
        related_d_v_index = list(set(self.related_d_v_index))
        return d_vectors[related_d_v_index], token_d_ids[related_d_v_index]




    def get_d_v_index(self, note):
        children = []
        new_hash_value = self.get_new_hash_value(note.layer, note.id)
        if note.is_leaf:
            self.related_d_v_index += note.vectors_indexes
        else:
            segment1 = LSHTreeIndex.split_binary_string(new_hash_value, 0, 4)
            segment2 = LSHTreeIndex.split_binary_string(new_hash_value, 2, 6)
            segment3 = LSHTreeIndex.split_binary_string(new_hash_value, 4, 8)
            children += list(note.binary_index[0][segment1])
            children += list(note.binary_index[1][segment2])
            children += list(note.binary_index[2][segment3])
            for child in children:
                self.get_d_v_index(child)


    def cal_hash_values(self, vector):
        self.hash_values  = np.tensordot(vector , self.lsh_database.hash_matrix, axes=([1],[2]))[0]
        self.hash_values = (self.hash_values > 0).astype(int)

    def get_new_hash_value(self, layer, id, hash_dim):
        return "".join(str(bit) for bit in self.hash_values[layer-1, id:id+hash_dim])


    @staticmethod
    def split_binary_string(binary_string, start_index, end_index):
        return binary_string[start_index:end_index]

    @staticmethod
    def init_binary_index(note, segment1, segment2, segment3):
        if segment1 not in list(note.binary_index[0].keys()):
            note.binary_index[0][segment1] = set()
        if segment2 not in list(note.binary_index[1].keys()):
            note.binary_index[1][segment2] = set()
        if segment3 not in list(note.binary_index[2].keys()):
            note.binary_index[2][segment3] = set()


