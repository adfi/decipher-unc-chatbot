import os
import pickle
import concurrent.futures

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from datetime import datetime

def embed_documents(chunks_path='data/chunks_20241221_180917.pkl'):
    datetime_str = '_'.join(chunks_path.split('_')[-2:]).split('.')[0]
    persist_directory = f"data/chroma_db_{datetime_str}"
    
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=FastEmbedEmbeddings(),
        persist_directory=persist_directory,  
    )
    
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    batches = [chunks[i:i + 100] for i in range(0, len(chunks), 100)]
  
    def process_batch(batch):
        print(datetime.now())
        vector_store.add_documents(batch)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        executor.map(process_batch, batches)
    vector_store.persist()

if __name__ == "__main__":
    embed_documents()