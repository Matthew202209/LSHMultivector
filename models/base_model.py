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
