import pickle
from datetime import datetime
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFDirectoryLoader


def chunk_documents():
    chunks = PyPDFDirectoryLoader("data/pdfdocs/").load_and_split()
    chunks = filter_complex_metadata(chunks)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"data/chunks_{current_datetime}.pkl", "wb") as f:
        pickle.dump(chunks, f)


if __name__ == "__main__":
    chunk_documents()
