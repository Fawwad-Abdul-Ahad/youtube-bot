import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embedding_model.embed_query("Hello world")

CHROMA_DIR = './chroma'
COLLECTION_NAME = 'youtube-transcripts'

chunks = [
    "Germany is a country in Central Europe.",
    "The capital of Germany is Berlin.",
    "German language has three genders: der, die, das.",
    "A1 level covers basic grammar and vocabulary.",
    "Nominative case is used for the subject of a sentence.",
]

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        "k" : 3,
        "fetch_k" : 10,
        "lambda_mult" : 0.5 
    }
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN")
)
print(os.getenv("HUGGINGFACE_TOKEN"))
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chat_history = []

prompt = ChatPromptTemplate(
    [
        (
            'system', """You are a helpful AI assistant that gives answer based on the context. 
            Context:
            {context}   
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            'user',
            "Question: {question}\nAnswer:"
        ),
    ]
)
query = "What is elbert einstein do"
def context_query (query):
    docs = retriever.invoke(query)
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

rag_chain = {"context" : RunnableLambda(context_query),"question": RunnablePassthrough(), "chat_history" : RunnableLambda(lambda x:chat_history)}  | prompt | model | parser
result = rag_chain.invoke(query)

chat_history.append(HumanMessage(content=query))
chat_history.append(AIMessage(content=result))

print(chat_history)
print(result)




