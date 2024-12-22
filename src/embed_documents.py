import pickle

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def embed_documents(chunks_path="data/chunks_20241221_180917.pkl"):
    datetime_str = "_".join(chunks_path.split("_")[-2:]).split(".")[0]
    persist_directory = f"data/chroma_db_{datetime_str}"

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en", encode_kwargs={"device": "cuda"}
    )

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    vector_store.add_documents(chunks)
    vector_store.persist()


if __name__ == "__main__":
    embed_documents()
