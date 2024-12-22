import pickle
import tqdm
from ChatOpenRouter import ChatOpenRouter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from src.secrets import OPENROUTER_API_KEY

def generate_questions():
    # Read the pickle file containing the list of Documents
    with open('data/chunks_20241221_180917.pkl', 'rb') as file:
        documents = pickle.load(file)
    
    llm = ChatOpenRouter(openai_api_key=OPENROUTER_API_KEY, model_name="openai/gpt-4o-2024-11-20")
    prompt = PromptTemplate.from_template("""You are a novel user of the Uniform Network Code (UNC) for the UK gas network. The following context is a snippet of the UNC. Create a question you may have about this text. Here are examples of questions (not necessarily based on the given text):

    'What does the peak day demand mean?'
    'What are the obligations upon the network operator for creating a demand forecast?'

    Only return the question. If you don't have a question, don't return anything.
Snippet:
{context}
                            """)
    chain = ({"context": RunnablePassthrough()}
                      | prompt
                      | llm
                      | StrOutputParser())
    
    responses = []
    for doc in tqdm.tqdm(documents[:200], desc="Generating questions"):
        responses.append(chain.invoke(doc.page_content))

    with open('data/questions_20241221_180917.pkl', 'wb') as file:
        pickle.dump(responses, file)

if __name__ == "__main__":
    generate_questions()