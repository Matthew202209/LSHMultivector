from distutils.command.build import build

import numpy as np

from models.lstm_app.lsh_database import LSHDatabase

class TreeNote:
    def __init__(self, layer, id):
        self.is_leaf = False
        self.layer = layer
        self.id = id
        self.children = []
        self.binary_index = []
        self.vectors_indexes = []




class LSHTreeIndex:
    def __init__(self, config, lsh_database:LSHDatabase):
        self.root = TreeNote(0,0)
        self.lsh_database = lsh_database
        self.tree_layers = config.tree_layers
        self.first_layer_hash_dim = config.first_layer_hash_dim
        self.hash_dim = config.hash_dim
        self.hash_values = None

    def build_tree(self):
    # first layer

        for id in range(2**self.first_layer_hash_dim):
            first_layer_note = TreeNote(1, id)
            self.build_other_layer(first_layer_note, 2, id)
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

    def insert(self, vector, vector_index):
        self.cal_hash_values(vector)
        hash_value = self.get_new_hash_value(0,0)
        self.set_binary_index(self.root, vector_index, hash_value = hash_value)

    def set_binary_index(self, note, vector_index, hash_value = None):
    # first layer
        if note.layer == 0:
            if len(note.binary_index) == 0:
                note.binary_index=[{}]

            note.binary_index[0][hash_value] = note.children[int(hash_value,2)]
            for child in note.children:
                self.set_binary_index(child, hash_value, vector_index)
        else:
            #todo setting as parameterization
            new_hash_value = self.get_new_hash_value(note.layer, note.id)
            if note.is_leaf:
                note.vectors_indexes.append(vector_index)
            else:
                segment1 = LSHTreeIndex.split_binary_string(new_hash_value,0,4)
                segment2 = LSHTreeIndex.split_binary_string(new_hash_value,2,6)
                segment3 = LSHTreeIndex.split_binary_string(new_hash_value,4,8)
                note.binary_index[0][segment1] = note.children[int(new_hash_value, 2)]
                note.binary_index[1][segment2] = note.children[int(new_hash_value, 2)]
                note.binary_index[2][segment3] = note.children[int(new_hash_value, 2)]
                self.set_binary_index(note.children[int(new_hash_value, 2)], vector_index)

    def search(self, vector):
        self.root.vectors_indexes.append(vector)

    def cal_hash_values(self, vector):
        matrix_3d = np.tile(vector, (self.tree_layers,
                                     2 ** (self.first_layer_hash_dim + self.hash_dim * (
                                                 self.tree_layers - 1)),
                                     1))

        self.hash_values = np.einsum("ijk,imn->ijm",self.lsh_database.hash_matrix , matrix_3d)

    def get_new_hash_value(self, layer, id):
        return "".join(str(bit) for bit in self.hash_values[layer, id, :])


    @staticmethod
    def split_binary_string(binary_string, start_index, end_index):
        return binary_string[start_index:end_index]


