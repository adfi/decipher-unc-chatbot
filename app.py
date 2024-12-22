from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from src.ChatOpenRouter import ChatOpenRouter
from src.config import OPENROUTER_API_KEY

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en"
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_20241212_180917",  
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenRouter(openai_api_key=OPENROUTER_API_KEY,model_name="meta-llama/llama-3.3-70b-instruct")

prompt = PromptTemplate.from_template(
"""
<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s> 
[INST] Question: {question} 
Context: {context} 
Answer: [/INST]
"""
)

chain = ({"context": retriever, "question": RunnablePassthrough()}
                      | prompt
                      | llm
                      | StrOutputParser())

