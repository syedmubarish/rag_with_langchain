from pathlib import Path
from typing import List,Any
from langchain_community.document_loaders import PyPDFLoader


def load_all_documents(data_dir:str) -> List[Any]:
    """
    Load pdf files from directory and convert it into langchain document data structure
    """

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []


    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")


    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to laod PDF {pdf_file}: {e}")

    return documents            

