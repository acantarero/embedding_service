from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from time import time

from src.sentence_chunker import SentenceChunker
from src.instructor import InstructorEmbedding
from src.embed import EmbeddedVector

from settings import INSTRUCTOR_MODEL

app = FastAPI()

# init embedding model globally at startup as it takes a while to load
embedding_model = InstructorEmbedding(INSTRUCTOR_MODEL)

class EmbeddingRequest(BaseModel):
    input: list[str]
    model: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

# TODO: heavy compute task, should be async
@app.post("/embeddings")
def embed(
    data: EmbeddingRequest, 
):
    # needed for instruction embedding models
    instruction = "represent the document for retrieval"

    vectors = []
    embedding_start = time()
    # match openai spec (for langchain compatibility):
    # https://platform.openai.com/docs/api-reference/embeddings/object
    for idx, v in enumerate(data.input):
        vectors.append({
            "object": "embedding",
            "embedding": EmbeddedVector(
                embedding_model.embed(v, instruction), v),
            "index": idx,
        })
    embedding_end = time()

    logger.info(f"embedding took {embedding_end - embedding_start} seconds for {len(data.input)} documents.")

    # match openai spec (for langchain compatibility):
    # https://platform.openai.com/docs/api-reference/embeddings/create
    return {
        "object": "list",
        "data": vectors,
        "model": data.model,
    } 
