from abc import ABC, abstractmethod
from collections import namedtuple

from numpy import ndarray

# document_key should be a unique identifier that can be generated from the source dataset
EmbeddedVector = namedtuple("EmbeddedVector", ["vector", "document_key", "text"])

class Embed(ABC):

    @abstractmethod    
    def embed(self) -> ndarray:
        """generate and return the embedding"""
        pass

