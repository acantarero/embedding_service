from enum import Enum
from loguru import logger

from InstructorEmbedding import INSTRUCTOR
from numpy import ndarray

from src.embed import Embed

class ModelSize(Enum):
    BASE = 1
    LARGE = 2
    XL = 3

class InstructorEmbedding(Embed):
    """Instructor embedding is an open source model that generates
    embeddings with instructions.  Details here:  

    https://instructor-embedding.github.io/

    3 model sizes are available on HuggingFace:

    - https://huggingface.co/hkunlp/instructor-xl
    - https://huggingface.co/hkunlp/instructor-large
    - https://huggingface.co/hkunlp/instructor-base
    """

    def __init__(self, 
                 size: ModelSize, 
                 document_instruction: str, 
                 query_instruction: str) -> None:

        logger.info("Initializing InstructorEmbedding")
        if size == ModelSize.BASE:
            self.model_name = 'hkunlp/instructor-base'
        elif size == ModelSize.LARGE:
            self.model_name = 'hkunlp/instructor-large'
            self.dimension = 768
        elif size == ModelSize.XL:
            self.model_name = 'hkunlp/instructor-xl'
        else:
            raise ValueError("Invalid model size.")
        
        self.model = INSTRUCTOR(self.model_name)
        logger.info(f"Initialized InstructorEmbedding with model: {self.model_name}")

        """Instructions to instructor models should include:
        - domain 
        - text_type
        - task_objective
            - for clustering
            - for retrieval
        """
        self.document_instruction = document_instruction
        self.query_instruction = query_instruction

    def embed(self, text: str) -> ndarray:
        value = self.model.encode([[self.document_instruction, text]])
        return value

    def embed_query(self, query: str) -> ndarray:
        result = self.model.encode([[self.query_instruction, query]])
        return result
