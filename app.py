from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import ChromaVectorStore

if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()

    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)

    chroma_store = ChromaVectorStore()
    # chroma_store.add_documents(chunks,embeddings)

    emb_pipe.embed_query("What is 307")
    