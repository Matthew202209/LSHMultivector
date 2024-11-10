from distutils.command.build import build

from models.lstm_app.lsh_database import LSHDatabase

class TreeNote:
    def __init__(self, layer, id):
        self.is_leaf = False
        self.layer = layer
        self.id = id
        self.children = []
        self.binary_index = None




class LSHTreeIndex:
    def __init__(self, config, lsh_database:LSHDatabase):
        self.root = TreeNote(0,0)
        self.lsh_database = lsh_database
        self.tree_layers = config.tree_layers
        self.first_layer_hash_dim = config.first_layer_hash_dim
        self.hash_dim = config.hash_dim

    def build_tree(self):
    # first layer

        for id in range(2**self.first_layer_hash_dim):
            first_layer_note = TreeNote(1, id)
            self.build_other_layer(first_layer_note, 2)
            self.root.children.append(first_layer_note)

    def build_other_layer(self, note, layer):
        for id in range(2**self.hash_dim):
            new_note = TreeNote(layer, id)
            if layer != self.tree_layers:
                self.build_other_layer(new_note, layer+1)
            else:
                new_note.is_leaf = True
            note.children.append(new_note)
