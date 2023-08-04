import unittest

from src.sentence_chunker import SentenceChunker

class TestChunker(unittest.TestCase):

    def test_no_chunking(self):
        input = "This text is very short.  It should not get chunked."
        expected = [input]
        chunker = SentenceChunker(max_size=1024)
        actual = chunker.chunk(input)

        self.assertCountEqual(actual, expected)

    def test_chunks_that_preserve_sentence_boundary(self):
        input = "A text that has chunks. They should break at sentence boundary.  Then we know it works."""
        expected = [
            "A text that has chunks.",
            "They should break at sentence boundary.",
            "Then we know it works.",
        ]
        chunker = SentenceChunker(max_size=40)
        actual = chunker.chunk(input)

        self.assertCountEqual(actual, expected)

    def test_whitespace_and_multiple_sentences_per_chunk(self):
        input = "Very short. Sentences.    With weird.    Punctuation.    Should chunk fine."
        expected = [
            "Very short. Sentences.",
            "With weird. Punctuation.",
            "Should chunk fine."
        ]
        chunker = SentenceChunker(max_size=25)
        actual = chunker.chunk(input)

        self.assertCountEqual(actual, expected)

    def test_sentence_too_long_for_chunker(self):
        input = "This sentence will not chunk correctly because it is too long.  This is short.  Also this."
        expected = [
            "This sentence will not chunk correctly",
            "because it is too long. This is short.",
            "Also this.",
        ]
        chunker = SentenceChunker(max_size=40)
        actual = chunker.chunk(input)

        self.assertCountEqual(actual, expected)

    def test_word_too_long(self):
        input = "Supercalifragilisticexpialidocious."
        expected = [
            "Supercalif",
            "ragilistic",
            "expialidoc",
            "ious.",
        ]
        chunker = SentenceChunker(max_size=10)
        actual = chunker.chunk(input)

        self.assertCountEqual(actual, expected)

    def test_word_too_long_mid_document(self):
        input = "This work is okay. Then we say supercalifragilisticexpialidocious which is long."
        expected = [
            "This work is okay.",
            "Then we say",
            "supercalifragilistic",
            "expialidocious",
            "which is long."
        ]
        chunker = SentenceChunker(max_size=20)
        actual = chunker.chunk(input)

        self.assertCountEqual(actual, expected)
