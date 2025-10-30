from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()

    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print(len(chunks))
    print(len(embeddings))