import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate

print("loading docs")
chunks = PyPDFDirectoryLoader("data/pdfdocs/").load_and_split()
chunks = filter_complex_metadata(chunks)
with open("chunks.pkl", 'wb') as f:
    pickle.dump(chunks, f)
print("embedding docs")

vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

print("initialising model")

llm = ChatOllama(model="mistral")

prompt = PromptTemplate.from_template(
"""
<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s> 
[INST] Question: {question} 
Context: {context} 
Answer: [/INST]
"""
)

print("creating chain")

chain = ({"context": retriever, "question": RunnablePassthrough()}
                      | prompt
                      | llm
                      | StrOutputParser())

print("generating answer")

answer = chain.invoke("What is the 1-in-20 peak demand?")
print(answer)