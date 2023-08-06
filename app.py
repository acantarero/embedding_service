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
    text: str

@app.get("/health")
def health_check():
    # TODO: check connection to DSE cluster is OK
    return {"status": "ok"}


# TODO: heavy compute task, should be async
#       option: put into pulsar/astra streaming write job to process stream
@app.post("/embed")
def embed(
    data: EmbeddingRequest, 
    instruction: str = "represent the document for retrieval",
):
    chunk_start = time()
    chunker = SentenceChunker(max_size=512)
    chunks = chunker.chunk(data.text)
    chunk_end = time()

    logger.info(f"chunking took {chunk_end - chunk_start} seconds")

    vectors = []
    embedding_start = time()
    for chunk in chunks:
        vectors.append(EmbeddedVector(embedding_model.embed(chunk, instruction),  
                                      chunk))
    embedding_end = time()

    logger.info(f"embedding took {embedding_end - embedding_start} seconds for {len(chunks)} chunks.")

    # assemble to return
    return_vectors = []
    for v in vectors:
        return_vectors.append({
            "vector": v.vector.tolist(),
            "text": v.text,
        })

    return {"results": return_vectors}
