from loguru import logger
import nltk

# TODO: move to env var in Dockerfile
nltk.data.path.append("data/nltk/")



class SentenceChunker:
    """A text chunking approach that preserves sentence boundaries
    and maximizes chunk size"""

    def __init__(self, max_size: int, overlap: float = 0.0):
        self.max_size = max_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        
        # TODO: add support for overlapping chunks
        if len(text) <= self.max_size:
            return [text]
        else:
            # NOTE: this approach only supports a subset of languages
            # NLTK website does not document well.  check NLTK download
            # directory to see available languages 
            sentences = nltk.sent_tokenize(text)

            text_chunk = ""
            chunks = []
            while len(sentences) > 0:
                if len(text_chunk) + len(sentences[0]) <= self.max_size:
                    text_chunk += sentences.pop(0) + ' '
                else:
                    if len(text_chunk) == 0: # sentence is too long
                        sentence = sentences.pop(0)
                        for word in sentence.split():
                            if len(text_chunk) + len(word) <= self.max_size:
                                text_chunk += word + ' '
                            else:
                                if len(text_chunk) > 0: # word is too long
                                    chunks.append(text_chunk.rstrip())
                                    text_chunk = ""

                                if len(word) > self.max_size:  
                                    chunks.extend([word[i:i+self.max_size] for i in range(0, len(word), self.max_size)])
                                else:
                                    text_chunk = word + ' ' 
                    else:
                        chunks.append(text_chunk.rstrip())
                        text_chunk = ""
            
            if len(text_chunk) > 0:
                chunks.append(text_chunk.rstrip())
 
        logger.debug(f"Chunked text into {len(chunks)} chunks.")
        return chunks


