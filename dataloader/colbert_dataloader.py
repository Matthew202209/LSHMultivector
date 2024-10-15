import ujson
from torch.utils.data import Dataset
from tqdm import tqdm


class ColbertDataset(Dataset):

    def __init__(self, path=None, corpus_dict=None):
        self.path = path
        self.corpus = corpus_dict or self._load_corpus(path)
        self.corpus_list =  list(self.corpus.values())
    def __len__(self):
        return len(self.corpus_list)

    def __getitem__(self, item):
        corpus_text =self.corpus_list[item]
        return corpus_text

    def _load_corpus(self, path):
        self.path = path
        corpus = {}
        num_lines = sum(1 for i in open(self.path, 'rb'))
        with open(self.path, encoding='utf-8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    corpus[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    corpus[line.get("doc_id")] = line.get("text")

        return corpus