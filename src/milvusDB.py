from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from embedder import Embedder
import numpy as np

class MilvusDB:
    def __init__(self, collection_name="semantic_search", emb_dim=384, host="localhost", port="19530"):
        self.collection_name = collection_name
        self.emb_dim = emb_dim
        self.model = Embedder()

        connections.connect("default", host=host, port=port)
        self._setup_collection()

    def _setup_collection(self):
        if self.collection_name in Collection.list():
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.emb_dim)
            ]
            schema = CollectionSchema(fields)
            self.collection = Collection(name=self.collection_name, schema=schema)

        self.collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "COSINE",
                          "index_type": "IVF_FLAT",
                          "params": {"nlist": 128}} # Number of clusters
        )

    def _ensure_loaded(self):
        if not self.collection.is_loaded:
            self.collection.load()

    # For table row
    def add_texts(self, texts: list[str]):
        embeddings = self.model.get_sentence_embedding(texts).tolist()
        self.collection.insert([texts, embeddings])

    def search(self, query: str, top_k=5, threshold=0.7):
        self._ensure_loaded()
        embedding = self.model.get_sentence_embedding(query).tolist()
        search_params = {"metric_type": "COSINE", 
                         "params": {"nprobe": 10}} # Explores only the clusters closest to the query vector.

        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )

        matches = []
        for match in results[0]:
            if match.score >= threshold:
                matches.append({
                    "text": match.entity.get("text"),
                    "score": match.score
                })
        return matches