import math
import os

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
    def __init__(self, config, token_reps_dim: int):
        self.config = config
        self.token_reps_dim = token_reps_dim
        self.hash_matrix = np.zeros((self.config.tree_layers-1,
                                     2**(self.config.first_layer_hash_dim+self.config.hash_dim*(self.config.tree_layers-3))*self.config.hash_dim,
                                     token_reps_dim))

        self.token_reps = []
        self.cls_reps = []
        self.token_d_ids = []


    def create_random_hash_matrix(self):
    # First layer
        self.hash_matrix[0, 0:self.config.first_layer_hash_dim,:] = create_random_hash_vectors(int(self.config.first_layer_hash_dim/2),
                                                                                                           self.token_reps_dim)
        for i in range(self.config.tree_layers-2):
            dim = 2**(self.config.first_layer_hash_dim+self.config.hash_dim*i)*self.config.hash_dim
            self.hash_matrix[i+1, 0:dim, :] = create_random_hash_vectors(int(dim/2), self.token_reps_dim)

    def save_index(self, save_path):
        np.save(os.path.join(save_path, "cls_reps.npy"), self.cls_reps)
        np.save(os.path.join(save_path, "token_reps.npy"), self.token_reps)
        np.save(os.path.join(save_path, "token_d_ids.npy"), self.token_d_ids)
        np.save(os.path.join(save_path, "hash_matrix.npy"), self.hash_matrix)

    def load_index(self, load_path):
        self.cls_reps = np.load(os.path.join(load_path, "cls_reps.npy"))
        self.token_reps  = np.load(os.path.join(load_path, "token_reps.npy"))
        self.token_reps = np.load(os.path.join(load_path, "token_d_ids.npy"))
        self.hash_matrix  = np.load(os.path.join(load_path, "hash_matrix.npy"))




def create_random_hash_vectors(num_vectors:int, token_reps_dim: int):
    hash_vectors = np.zeros((num_vectors*2, token_reps_dim))
    for i in range(num_vectors):
        v1 = np.random.choice([-1, 1], size=token_reps_dim)/math.sqrt(token_reps_dim)
        v2 = find_orthogonal_unit_vector(v1, token_reps_dim)
        hash_vectors[2 * i] = v1
        hash_vectors[2 * i +1] = v2
    A = np.random.randn(token_reps_dim, token_reps_dim)

    # 进行QR分解
    Q, _ = np.linalg.qr(A)
    return hash_vectors @ Q

def find_orthogonal_unit_vector(u, token_reps_dim):
    # 确保u是一个numpy数组
    u = np.array(u)
    # 生成一个随机的128维向量
    v = np.random.rand(token_reps_dim)
    # 标准化v
    v = v / np.linalg.norm(v)
    # 计算v在u上的投影
    projection = np.dot(v, u) * u
    # 从v中减去投影，得到正交向量
    v_orthogonal = v - projection
    # 标准化正交向量
    v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal)
    return v_orthogonal