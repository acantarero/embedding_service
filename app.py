from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from time import time

from src.sentence_chunker import SentenceChunker
from src.instructor import InstructorEmbedding, ModelSize
from src.embed import EmbeddedVector

app = FastAPI()

# init embedding model globally at startup as it takes a while to load
# since document and que
embedding_model = InstructorEmbedding(
    ModelSize.XL,
    document_instruction="represent the news document for retrieval",
    query_instruction="represent the news headline for retrieving support documents",
)


class EmbeddingRequest(BaseModel):
    text: str
    document_id: str
    metadata: dict


@app.get("/health")
def health_check():
    # TODO: check connection to DSE cluster is OK
    return {"status": "ok"}


# TODO: heavy compute task, should be async
#       option: put into pulsar/astra streaming write job to process stream
@app.post("/embed")
def embed(data: EmbeddingRequest):
    chunk_start = time()
    chunker = SentenceChunker(max_size=512)
    chunks = chunker.chunk(data.text)
    chunk_end = time()

    logger.info(f"chunking took {chunk_end - chunk_start} seconds")

    vectors = []
    embedding_start = time()
    for chunk in chunks:
        vectors.append(EmbeddedVector(embedding_model.embed(chunk), data.document_id, chunk))
    embedding_end = time()

    logger.info(f"embedding took {embedding_end - embedding_start} seconds for {len(chunks)} chunks.")

     # write to DSE
    # return number of chunks written to DSE 
    return {"status": "ok"}
