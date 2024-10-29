import torch
import torch_scatter
from tqdm import tqdm
from bitstring import BitArray
from models.Hamming.hamming_search import HammingSearcher


class HammingAccSearcher(HammingSearcher):
    def __init__(self, config, num_doc):
        super().__init__(config, num_doc)

    def prepare_index(self):
        index_dir = r"{}/index/{}/hamming_acc/{}".format(self.config.save_dir, self.config.dataset,
                                                         str(self.config.hash_dimmension))
        self.cls_reps = torch.load(r"{}/{}".format(index_dir, 'cls_reps.pt'), map_location="cpu")
        self.token_labels = torch.load(r"{}/{}".format(index_dir, 'token_labels.pt'), map_location="cpu")
        self.hash_matrix = torch.load(r"{}/{}".format(index_dir, 'hash_matrix.pt'), map_location="cpu")
        self.hash_bins = torch.load(r"{}/{}".format(index_dir, 'hash_bins.pt'), map_location="cpu")


    def lsh_search(self, embeddings):
        hamming_matrix = self.cal_hash_value(embeddings).squeeze()
        for i, hamming_key in enumerate(tqdm(hamming_matrix)):
            hash_value = "".join(hamming_key.numpy().astype(str))
            doc_token_reps, token_pid = self.find_similar_bins(hash_value)
            token_score = torch.matmul(embeddings[0][i], doc_token_reps.T).relu_()
            token_pid_tensor = torch.Tensor(token_pid).to(torch.int64)
            torch_scatter.scatter_max(src=token_score, index=token_pid_tensor, out=self.max_scores, dim=-1)
            self.sum_scores += self.max_scores
            self.max_scores.fill_(0)


    def find_similar_bins(self, query):
        doc_token_reps = []
        token_pid = []
        for key, value in self.hash_bins.items():
            if HammingAccSearcher.hamming_distance(query, key) <= self.hamming_threshold:
                doc_token_reps.append(value["dense_repr"][0])
                token_pid += value["dense_repr"][1]
        doc_token_reps = torch.cat(doc_token_reps, dim=0)
        return doc_token_reps, token_pid

    @staticmethod
    def hamming_distance(bits1,bits2):
        bits1 = "0b" + bits1
        bits2 = "0b" + bits2
        b1 = BitArray(bits1)
        b2 = BitArray(bits2)
        xor_result = b1 ^ b2

        # 计算结果中'1'的数量，即汉明距离
        distance = xor_result.count('1')

        return distance