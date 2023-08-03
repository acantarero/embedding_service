from enum import Enum
from loguru import logger

from InstructorEmbedding import INSTRUCTOR
from numpy import ndarray

from src.embed import Embed


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
                 model: str = "hkunlp/instructor-base", 
                 document_instruction: str = None, 
                 query_instruction: str = None) -> None:
        """Leave option to globally initialize with instructions
        """

        self.model_name = model
        self.dimension = 768

        logger.info("Initializing InstructorEmbedding")
        if self.model_name not in ["hkunlp/instructor-base", "hkunlp/instructor-large", "hkunlp/instructor-xl"]:
            raise ValueError("Invalid model. Must be one of 'hkunlp/instructor-base', 'hkunlp/instructor-large', 'hkunlp/instructor-xl'.")

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

    def embed(self, text: str, document_instruction: str = None) -> ndarray:
        value = self.model.encode([[document_instruction, text]])
        return value

    def embed_query(self, query: str, query_instruction: str = None) -> ndarray:
        result = self.model.encode([[query_instruction, query]])
        return result
