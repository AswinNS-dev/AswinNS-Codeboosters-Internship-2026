TOP_K = 5

class RAGPipeline:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def retrieve(self, query, top_k=TOP_K):
        collection = self.kb.get_collection()
        embedder = self.kb.get_embedder()

        # check if collection has any data
        try:
            count = collection.count()
        except Exception:
            count = 0

        if count == 0:
            return "No knowledge base available yet. Answering from general knowledge."

        query_embedding = embedder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return "No relevant context found. Answering from general knowledge."

        context_parts = []
        for doc, meta, dist in zip(docs, metas, distances):
            similarity = round(1 - dist, 3)
            context_parts.append(
                f"[Category: {meta.get('category', 'N/A')} | "
                f"Subcategory: {meta.get('subcategory', 'N/A')} | "
                f"Relevance: {similarity}]\n{doc}"
            )

        return "\n\n---\n\n".join(context_parts)
