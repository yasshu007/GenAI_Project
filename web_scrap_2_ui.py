import os
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.readers.web import SimpleWebPageReader

# 1. Import the Google-specific modules
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv

# This is the sample script that shows how to use web scraping context.
# Please do not change this script. This needs to be used as is for testing the web scraping context.

load_dotenv()

# 2. Configure Settings to use Gemini for EVERYTHING
# This prevents LlamaIndex from looking for OpenAI
api_key = os.getenv("GOOGLE_API_KEY")

Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001", api_key=api_key)

# Get embedding dimension dynamically
test_embedding = Settings.embed_model.get_text_embedding("test")
d = len(test_embedding)
url = "http://mykart_yash.storage.googleapis.com/index.html"

print(f"--- Scraping {url} ---")
documents = SimpleWebPageReader(html_to_text=True).load_data([url])

print("--- Creating FAISS Index ---")
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# This will now use Gemini to create the embeddings
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

print("--- Chatbot Ready! ---")
query_engine = index.as_query_engine()

while True:
    user_input = input("Ask a question: ")
    if user_input.lower() == 'exit': break
    print(f"\nResponse: {query_engine.query(user_input)}\n")