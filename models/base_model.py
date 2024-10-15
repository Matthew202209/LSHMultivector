from abc import ABC, abstractmethod

class BaseIndex(ABC):
    def __init__(self, config):
        self.config = config
        self.context_encoder = None
        self.dataset = None

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def indexing(self):
        pass

    @abstractmethod
    def save_index(self):
        pass

class BaseRetrieve(ABC):
    def __init__(self, config):
        self.config = config
        self.searcher = None
        self.context_encoder = None
        self.labels = None
        self.queries = None
        self.corpus = None

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def prepare_searcher(self):
        pass


    @abstractmethod
    def retrieve(self):
        pass

    @abstractmethod
    def evaluation(self, path):
        pass

    @abstractmethod
    def save_ranks(self):
        pass


class BaseSearcher(ABC):
    def __init__(self, config):
        self.config = config


    @abstractmethod
    def prepare_index(self):
        pass

    @abstractmethod
    def search(self, q_repr):
        pass

