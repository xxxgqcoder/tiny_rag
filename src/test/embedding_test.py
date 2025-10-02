from common.config import TinyRAGConfig
from rag.embedding import BGEEmbedding


class TestBGEEmbedding:
    def test_bge_embedding_initialization(self):
        """Test BGEEmbedding model initialization"""
        model = BGEEmbedding(model_name="bge-m3:567m")
        assert model.model_name == "bge-m3:567m"
        assert model.client is not None

    def test_bge_embedding_encode_single_text(self):
        """Test encoding a single text with BGE model"""
        model = BGEEmbedding(model_name="bge-m3:567m")
        texts = ["Hello world"]
        embeddings = model.encode(texts)

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        assert all(isinstance(x, float) for x in embeddings[0])

    def test_bge_embedding_encode_multiple_texts(self):
        """Test encoding multiple texts with BGE model"""
        model = BGEEmbedding(model_name="bge-m3:567m")
        texts = ["Hello world", "Python programming", "Machine learning"]
        embeddings = model.encode(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        # Check all embeddings have the same dimension
        assert len(set(len(emb) for emb in embeddings)) == 1
        assert len(embeddings[0]) == TinyRAGConfig.embedding_config.embedding_dim

    def test_bge_embedding_key_generator(self):
        """Test key generation for caching"""
        model = BGEEmbedding(model_name="bge-m3:567m")
        texts = ["test text"]
        key1 = model.key_generator(texts)
        key2 = model.key_generator(texts)

        assert key1 == key2
        assert key1.startswith("embedding::text_hash::")

        # Different texts should generate different keys
        key3 = model.key_generator(["different text"])
        assert key1 != key3

    def test_bge_embedding_caching(self):
        """Test that caching works for repeated calls"""
        model = BGEEmbedding(model_name="bge-m3:567m")
        texts = ["Caching test text"]

        embeddings1 = model.encode(texts)
        embeddings2 = model.encode(texts)

        assert embeddings1 == embeddings2

    def test_bge_embedding_singleton(self):
        """Test that BGEEmbedding follows singleton pattern"""
        model1 = BGEEmbedding(model_name="bge-m3:567m")
        model2 = BGEEmbedding(model_name="bge-m3:567m")

        assert model1 is model2
