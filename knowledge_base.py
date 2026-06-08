import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "healthcare_kb"

class KnowledgeBase:
    def __init__(self, snowflake_store):
        self.store = snowflake_store
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_text(self, text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def _clean(self, text):
        return " ".join(text.strip().split())

    def build(self):
        interactions = self.store.fetch_all_interactions()
        if not interactions:
            return 0

        # delete old collection and recreate for fresh rebuild
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        documents, embeddings, metadatas, ids = [], [], [], []
        chunk_id = 0

        for interaction in interactions:
            combined = (
                f"Q: {self._clean(interaction['question'])}\n"
                f"A: {self._clean(interaction['response'])}"
            )
            chunks = self._chunk_text(combined)

            for chunk in chunks:
                embedding = self.embedder.encode(chunk).tolist()
                documents.append(chunk)
                embeddings.append(embedding)
                metadatas.append({
                    "category": interaction["category"],
                    "subcategory": interaction["subcategory"],
                })
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

        # batch upsert
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )

        return chunk_id

    def get_collection(self):
        return self.collection

    def get_embedder(self):
        return self.embedder
