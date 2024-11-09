import math

import numpy as np

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class LSHDatabase:
    def __init__(self, config, num_token: int, token_reps_dim: int):
        self.config = config
        self.num_token = num_token
        self.token_reps_dim = token_reps_dim
        self.token_reps = np.zeros((num_token, token_reps_dim))
        self.hash_matrix = np.zeros((self.config.tree_layers,
                                     2**(self.config.first_layer_hash_dim+self.config.hash_dim*(self.config.tree_layers-1)),
                                     token_reps_dim))


    def create_random_hash_matrix(self):
    # First layer
        self.hash_matrix[0, 0:self.config.first_layer_hash_dim,:] = LSHDatabase.create_random_hash_vectors(self.config.first_layer_hash_dim,
                                                                                                           self.token_reps_dim)
        for i in range(self.config.tree_layers-1):
            dim = 2**(self.config.first_layer_hash_dim + (i+1)*self.config.hash_dim)-1
            self.hash_matrix[i, 0:dim, :] = LSHDatabase.create_random_hash_vectors(self.config.hash_dim/2, self.token_reps_dim)

    @staticmethod
    def create_random_hash_vectors(num_vectors:int, token_reps_dim: int):
        hash_vectors = np.zeros((num_vectors*2, token_reps_dim))
        for i in range(num_vectors):
            v1 = np.random.choice([-1, 1], size=token_reps_dim)/math.sqrt(token_reps_dim)
            v2 = np.cross(np.random.rand(token_reps_dim), v1)
            v2 = v2/np.linalg.norm(v2)
            hash_vectors[2 * i] = v1
            hash_vectors[2 * i +1] = v2
        A = np.random.randn(token_reps_dim, token_reps_dim)

        # 进行QR分解
        Q, _ = np.linalg.qr(A)

        return hash_vectors @ Q