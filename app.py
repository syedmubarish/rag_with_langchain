from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import ChromaVectorStore
from src.chroma_retriever import ChromaRetriever
from src.search import do_rag,llm

if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()

    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)

    chroma_store = ChromaVectorStore()
    # chroma_store.add_documents(chunks,embeddings) --> Uncomment this if chunks isnt added to vector db

    chroma_retriever = ChromaRetriever(chroma_store,emb_pipe)
    

    output=do_rag("What is punishment for theft",chroma_retriever,llm,summarization=True)
    print(output)
    