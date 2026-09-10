from langchain_vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import os

def vector_process(video_id : str): 