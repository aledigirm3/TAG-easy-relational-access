from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

class MilvusDB:
    def __init__(self, embedder, collection_name="TAG_search", host="localhost", port="19530"):
        self.collection_name = collection_name
        self.model = embedder
        self.emb_dim = self.model.get_embedding_dimension()

        connections.connect("default", host=host, port=port)
        self._setup_collection()

    def _setup_collection(self):
        collections = utility.list_collections()
        if self.collection_name in collections:
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
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
        print(f"Milvus collection -> {self.collection}")

    def _ensure_loaded(self):
            self.collection.load()

    # For table row
    def add_texts(self, texts: list[str], batch_size: int = 1000):
        embeddings = self.model.get_sentence_embedding(texts).tolist()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]

            self.collection.insert([batch_texts, batch_embeddings])

    def search(self, query: str, threshold=0.3, top_k=10000):
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