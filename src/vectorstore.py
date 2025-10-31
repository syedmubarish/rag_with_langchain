import os
import chromadb
import numpy as np
from typing import List,Any
from src.embedding import EmbeddingPipeline
from sentence_transformers import SentenceTransformer
import uuid


class ChromaVectorStore:
    def __init__(self,collection_name="pdf_documents_ipc",persist_directory="db/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"PDF document embeddings for RAG"}
            )
            print(f"[INFO] Vector store initialised. Collection:{self.collection_name}")
            print(f"[INFO] Existing documents in collection:{self.collection.count()}")

        except Exception as e:
            print(f"[ERROR]Error initializing vector store: {e}")
    
    def add_documents(self,chunks:List[Any], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match embeddings")
        
        print(f"[INFO] Adding {len(embeddings)} to vector store....")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc,embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )

            print(f"[INFO]Successfully added {len(chunks)} documents to vector store")
            print(f"[INFO]Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"[ERROR] Adding documents to chromadb : {e}")
            raise

