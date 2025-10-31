from src.embedding import EmbeddingPipeline
from src.vectorstore import ChromaVectorStore
from typing import List, Dict, Any


class ChromaRetriever:
    def __init__(
        self, vector_store: ChromaVectorStore, embedding_pipeline: EmbeddingPipeline
    ):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        print(f"[INFO]Retrieving documents for query: '{query}'")
        print(f"[INFO]Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_pipeline.embed_query(query)

        try:
            results = self.vector_store.collection.query(
                query_embeddings=query_embedding.tolist(), n_results=top_k
            )

            retrieved_docs = []
            if results["documents"]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance
                    

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )
                print(
                    f"[INFO]Retrieved {len(retrieved_docs)} documents after filtering"
                )
            else:
                print(f"[INFO] No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"[ERROR]Error during retrievel: {e}")
            return []
