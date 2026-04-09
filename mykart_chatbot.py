# MyKart Online Store Assistant
# This script is working with all the scenarios: 1. DB data ingestion, 2. Web scraping, 3. PDF ingestion - all in one place. This is the main script to run for the final demo.
import os
import json
import base64
import faiss
import streamlit as st
import hashlib
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.readers.web import SimpleWebPageReader

# 1. Import the Google-specific modules
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.node_parser import LangchainNodeParser

# Fix for Streamlit's threaded environment
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()



# ── Constants ─────────────────────────────────────────────────────────────────
FAISS_INDEX_DIR = "faiss_store"
COMBINED_INDEX  = os.path.join(FAISS_INDEX_DIR, "combined_index")
REGISTRY_FILE   = os.path.join(FAISS_INDEX_DIR, "registry.json")
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# ── MyKart Brand Colors ───────────────────────────────────────────────────────
BRAND_ORANGE  = "#E8500A"   # from logo background
BRAND_DARK    = "#3A3A3A"   # dark charcoal from logo text
BRAND_YELLOW  = "#F5C518"   # bee wing gold
BRAND_LIGHT   = "#FFF8F3"   # warm off-white
BRAND_ORANGE2 = "#FF6B2B"   # lighter orange accent

# ── SVG Avatars (inline, no external deps) ───────────────────────────────────
# User avatar: professional person silhouette in MyKart dark/orange
USER_AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" class="user-avatar">
  <rect width="96" height="96" x="2" y="2" rx="28" fill="#2D3436" stroke="#3A3A3A" stroke-width="2"/>
  
  <path d="M30 2 C 10 2, 2 10, 2 30" fill="none" stroke="white" stroke-width="1" stroke-opacity="0.1"/>

  <circle cx="50" cy="38" r="14" fill="#FFD700"/>
  
  <path d="M25 80 Q 25 60 50 60 Q 75 60 75 80" fill="none" stroke="#FFD700" stroke-width="12" stroke-linecap="round"/>
  
  <circle cx="80" cy="80" r="8" fill="#E8500A" stroke="#2D3436" stroke-width="3"/>
</svg>
"""

# Assistant avatar: stylized basket in MyKart orange
ASSISTANT_AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" class="bot-avatar">
  <circle cx="50" cy="50" r="48" fill="#E8500A" />
  
  <path d="M25 40 Q 50 15 75 40" fill="none" stroke="#FFD700" stroke-width="6" stroke-linecap="round" class="basket-handle"/>
  
  <path d="M15 40 H 85 L 78 85 H 22 Z" fill="#FFFFFF" stroke="#2D3436" stroke-width="2" class="basket-body"/>
  
  <circle cx="35" cy="55" r="4" fill="#FFD700"/>
  <circle cx="50" cy="55" r="4" fill="#E8500A"/>
  <circle cx="65" cy="55" r="4" fill="#FFD700"/>
  
  <circle cx="42" cy="70" r="3" fill="#2D3436"/>
  <circle cx="58" cy="70" r="3" fill="#2D3436"/>
</svg>
"""

def svg_to_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.strip().encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"

USER_AVATAR      = svg_to_data_uri(USER_AVATAR_SVG)
ASSISTANT_AVATAR = svg_to_data_uri(ASSISTANT_AVATAR_SVG)

# ── MyKart Logo SVG (for header) ─────────────────────────────────────────────
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" class="site-shopping-logo">
  <circle cx="50" cy="50" r="48" fill="#2D3436" stroke="#3A3A3A" stroke-width="2"/>
  
  <circle cx="42" cy="-10" r="5" fill="#FFD700" class="logo-drop-item logo-item-1" />
  
  <circle cx="58" cy="-10" r="5" fill="#E8500A" class="logo-drop-item logo-item-2" />

  <path d="M30 40 Q 50 10 70 40" fill="none" stroke="#FFD700" stroke-width="5" stroke-linecap="round" class="logo-basket-handle"/>
  
  <path d="M20 40 H 80 L 72 80 H 28 Z" fill="#FFFFFF" class="logo-basket-body" />
  
  <rect x="25" y="50" width="50" height="4" rx="2" fill="#FFD700" opacity="0.9" />
</svg>
"""

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = f"""
<style>
  /* ── Import font ── */
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

  /* ── Root / app background ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {BRAND_LIGHT} !important;
      font-family: 'Nunito', sans-serif !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background: linear-gradient(160deg, {BRAND_DARK} 0%, #1e1e1e 100%) !important;
      border-right: 3px solid {BRAND_ORANGE} !important;
  }}
  [data-testid="stSidebar"] * {{
      color: #f0f0f0 !important;
      font-family: 'Nunito', sans-serif !important;
  }}
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{
      color: {BRAND_YELLOW} !important;
      font-weight: 800 !important;
  }}
  [data-testid="stSidebar"] hr {{
      border-color: {BRAND_ORANGE} !important;
      opacity: 0.4;
  }}

  /* ── Sidebar file uploader & buttons ── */
  [data-testid="stSidebar"] .stButton > button {{
      background: {BRAND_ORANGE} !important;
      color: white !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
      font-family: 'Nunito', sans-serif !important;
      transition: all 0.2s ease;
  }}
  [data-testid="stSidebar"] .stButton > button:hover {{
      background: {BRAND_ORANGE2} !important;
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(232,80,10,0.4) !important;
  }}

  /* ── Main header area ── */
  .mykart-header {{
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 18px 24px;
      background: linear-gradient(135deg, {BRAND_DARK} 0%, #2a2a2a 100%);
      border-radius: 16px;
      margin-bottom: 20px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.18);
      border-left: 5px solid {BRAND_ORANGE};
  }}
  .mykart-header-text h1 {{
      margin: 0;
      font-size: 1.7rem;
      font-weight: 800;
      color: white;
      font-family: 'Nunito', sans-serif;
      line-height: 1.2;
  }}
  .mykart-header-text h1 span {{
      color: {BRAND_YELLOW};
  }}
  .mykart-header-text p {{
      margin: 4px 0 0 0;
      font-size: 0.82rem;
      color: #aaa;
      font-family: 'Nunito', sans-serif;
  }}

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {{
      background: white !important;
      border-radius: 14px !important;
      padding: 14px 18px !important;
      margin-bottom: 10px !important;
      box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important;
      border: 1px solid #f0e8e0 !important;
      font-family: 'Nunito', sans-serif !important;
  }}
  /* User messages: warm tinted background */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
  [data-testid="stChatMessage"]:nth-child(odd) {{
      background: #fff5ef !important;
      border-color: #f5d5c0 !important;
  }}

  /* ── Sidebar file uploader — dark bg so text is readable ── */
  [data-testid="stSidebar"] [data-testid="stFileUploader"] {{
      background: rgba(255,255,255,0.08) !important;
      border: 1px dashed rgba(245,197,24,0.5) !important;
      border-radius: 10px !important;
      padding: 8px !important;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
      color: #f0f0f0 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
  [data-testid="stSidebar"] [data-testid="stFileUploader"] span {{
      color: #cccccc !important;
  }}
  /* Browse files button specifically */
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
      background: {BRAND_YELLOW} !important;
      color: {BRAND_DARK} !important;
      border: none !important;
      font-weight: 700 !important;
      border-radius: 6px !important;
  }}

  /* ── Sidebar knowledge base code/monospace text ── */
  [data-testid="stSidebar"] code {{
      background: rgba(255,255,255,0.12) !important;
      color: {BRAND_YELLOW} !important;
      border-radius: 4px !important;
      padding: 1px 5px !important;
  }}

  /* ── Chat input — remove orange border, keep clean default ── */
  [data-testid="stChatInput"] textarea {{
      background: white !important;
      border: none !important;
      border-radius: 12px !important;
      font-family: 'Nunito', sans-serif !important;
      font-size: 0.95rem !important;
      color: {BRAND_DARK} !important;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
  }}
  [data-testid="stChatInput"] textarea:focus {{
      box-shadow: 0 2px 12px rgba(0,0,0,0.12) !important;
      outline: none !important;
  }}
  [data-testid="stChatInput"] button {{
      background: {BRAND_ORANGE} !important;
      border-radius: 8px !important;
  }}

  /* ── Info / warning boxes ── */
  [data-testid="stInfo"] {{
      background: #fff5ef !important;
      border-left-color: {BRAND_ORANGE} !important;
      border-radius: 10px !important;
      font-family: 'Nunito', sans-serif !important;
  }}

  /* ── Spinner ── */
  [data-testid="stSpinner"] p {{
      color: {BRAND_ORANGE} !important;
      font-family: 'Nunito', sans-serif !important;
  }}

  /* ── Hide default Streamlit title so our custom header shows ── */
  .stApp > header {{ display: none; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {BRAND_LIGHT}; }}
  ::-webkit-scrollbar-thumb {{ background: {BRAND_ORANGE}; border-radius: 10px; }}
</style>
"""

# ── Models ────────────────────────────────────────────────────────────────────
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
)

Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))



# ── Registry Helpers ──────────────────────────────────────────────────────────
def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry: dict) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

# ── Core RAG Helpers ──────────────────────────────────────────────────────────
# function to extract text from PDF using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_file) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Function to compute MD5 hash of text for deduplication
def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# Function to chunk text using RecursiveCharacterTextSplitter
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)


# 2. Setup the Recursive Character Splitter
# This splitter tries to keep paragraphs, then sentences, then words together.
lc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

# Wrap it so LlamaIndex can use it
Settings.node_parser = LangchainNodeParser(lc_splitter)



# function to ingest PDF: 
# Extract -> Hexadecimal fingerprint -> Deduplicate -> Embeddings creation -> Merge into combined FAISS index
def ingest_pdf(pdf_file) -> tuple[bool, str]:
    """
    Incremental RAG ingestion:
    Extract → fingerprint → deduplicate → embed → merge into combined FAISS index.
    """
    registry = load_registry()
    text     = extract_text_from_pdf(pdf_file)
    md5_hash = compute_md5(text)

    if md5_hash in registry:
        return False, f"⚠️ **{pdf_file.name}** was already ingested. Skipping duplicates."

    chunks    = chunk_text(text)
    new_store = FAISS.from_texts(chunks, embedding_model)

    if os.path.exists(COMBINED_INDEX):
        combined = FAISS.load_local(
            COMBINED_INDEX, embedding_model, allow_dangerous_deserialization=True
        )
        combined.merge_from(new_store)   # ← incremental merge
    else:
        combined = new_store

    combined.save_local(COMBINED_INDEX)
    registry[md5_hash] = pdf_file.name
    save_registry(registry)

    return True, f"✅ **{pdf_file.name}** ingested — {len(chunks)} chunks added to KB."

def web_scrap():

    """
    Incremental RAG ingestion:
    Extract → fingerprint → deduplicate → embed → merge into combined FAISS index.
    """

    # Get embedding dimension dynamically
    test_embedding = Settings.embed_model.get_text_embedding("test")
    d = len(test_embedding)
    url = "http://mykart_yash.storage.googleapis.com/index.html"

    print(f"--- Scraping {url} ---")
    documents = SimpleWebPageReader(html_to_text=True).load_data([url])
    print("--- Creating FAISS Index ---")
    texts = [doc.text for doc in documents]
    new_store = FAISS.from_texts(texts, embedding_model)

    if os.path.exists(COMBINED_INDEX):
        combined = FAISS.load_local(
            COMBINED_INDEX, embedding_model, allow_dangerous_deserialization=True
        )
        combined.merge_from(new_store)
    else:
        combined = new_store

    combined.save_local(COMBINED_INDEX)
    print("--- FAISS Index Created ---")
    return True, f"✅ Web page data scraping done... Added to KB."


# Function to load the combined FAISS index (if it exists)
def load_combined_index():
    if os.path.exists(COMBINED_INDEX):
        return FAISS.load_local(
            COMBINED_INDEX, embedding_model, allow_dangerous_deserialization=True
        )
    return None

# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(context: str, history: list[dict], question: str) -> list[dict]:
    system_message = {
        "role": "system",
        "content": (
            "You are the MyKart Online Store Assistant — an expert AI helpdesk agent in answering questions about the MyKart online store, its products and also data about historical purchases. "
            "You need to answer only using the provided document context and data sources only. "
            "Be concise, warm, encouraging, and professional — in the spirit of MyKart Online Store. "
            "If the answer is not in the context or data sources, say so honestly and suggest reaching out to the MyKart team without any other thought."
        ),
    }
    history_messages = []
    for turn in history:
        history_messages.append({"role": "user",      "content": turn["question"]})
        history_messages.append({"role": "assistant", "content": turn["answer"]})

    user_message = {
        "role": "user",
        "content": (
            f"Use the following document excerpts to answer the question.\n\n"
            f"--- Document Context ---\n{context}\n--- End Context ---\n\n"
            f"Question: {question}"
        ),
    }
    return [system_message] + history_messages + [user_message]


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MyKart Online Store Assistant",
    page_icon="🛒",
    layout="centered"
)

# Inject CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Custom branded header ─────────────────────────────────────────────────────
logo_b64 = base64.b64encode(LOGO_SVG.strip().encode()).decode()
st.markdown(f"""
<div class="mykart-header">
    <img src="data:image/svg+xml;base64,{logo_b64}" width="140" alt="MyKart Logo"/>
    <div class="mykart-header-text">
        <h1>Online Store<span> Assistant</span></h1>
        <p> 🛒 Your AI powered assistant &nbsp; &nbsp; &nbsp; MyKart Store</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db" not in st.session_state:
    st.session_state.db = load_combined_index()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Sidebar logo
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 4px 0;">
        <img src="data:image/svg+xml;base64,{logo_b64}" width="180" alt="MyKart"/>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Upload Documents")
    st.caption("Each PDF is merged into the shared KB.")
 
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.button("➕ Ingest into KB"):
            with st.spinner("🛒 Embedding and merging…"):
                ingested, message = ingest_pdf(uploaded_file)
            st.markdown(message)
            if ingested:
                st.session_state.db = load_combined_index()
    st.markdown("---")
    st.markdown("### 📄 Webscraping Data from page")
    st.caption("Each page is merged into the shared KB.")
 
    if st.button("➕ Webscraping into KB"):
        with st.spinner("🛒 Embedding and merging…"):
            ingested, message = web_scrap()
        st.markdown(message)
        if ingested:
            st.session_state.db = load_combined_index()
    
        st.markdown("✅ Webscraping data ingested into KB.")

    st.markdown("---")
    st.caption("Ingesting Sales data from SQL to KB.")
    if st.button("🗑️ Ingest Sales Data to KB"):
        from sql_data_exec import extract_and_prepare_data    
        df = extract_and_prepare_data('mykart.db')
        # Convert the 'text_to_embed' column to a list of strings for FAISS ingestion   
        texts = df['text_to_embed'].tolist()
        new_store = FAISS.from_texts(texts, embedding_model)
        if os.path.exists(COMBINED_INDEX):
            combined = FAISS.load_local(
                COMBINED_INDEX, embedding_model, allow_dangerous_deserialization=True
            )
            combined.merge_from(new_store)   # ← incremental merge
            combined.save_local(COMBINED_INDEX)
        else:
            new_store.save_local(COMBINED_INDEX)    
        st.session_state.db = load_combined_index()

        st.markdown("✅ Sales data ingested into KB.")

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    registry = load_registry()
    if registry:
        for i, (_, fname) in enumerate(registry.items(), 1):
            st.markdown(f"&nbsp;&nbsp;📄 `{fname}`")
        st.caption(f"**{len(registry)}** document(s) loaded")
    else:
        st.caption("No documents ingested yet.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Reset KB"):
            import shutil # file and directory operations module
            if os.path.exists(COMBINED_INDEX): shutil.rmtree(COMBINED_INDEX)
            if os.path.exists(REGISTRY_FILE):  os.remove(REGISTRY_FILE)
            st.session_state.db           = None
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("💬 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    for q in [
        "How to reach out to the MyKart team?",
        "What is the base location of MyKart?",
        "What are the inventory types from MyKart?",
        "What are payment modes accepted by MyKart Store?",
    ]:
        st.markdown(f"&nbsp;&nbsp;› {q}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.75rem;'>"
        "©2026 MyKart Academy<br/>All rights reserved</div>",
        unsafe_allow_html=True
    )

# ── Main Chat Area ────────────────────────────────────────────────────────────
if not st.session_state.db:
    st.markdown("""
    <div style="
        background: white;
        border: 2px dashed #E8500A;
        border-radius: 16px;
        padding: 36px;
        text-align: center;
        color: #3A3A3A;
        font-family: 'Nunito', sans-serif;
        margin-top: 20px;
    ">
        <div style="font-size: 3rem;">�</div>
        <h3 style="color: #E8500A; margin: 10px 0 6px 0;">Welcome to MyKart Online Store Assistant</h3>
        <p style="color: #666; margin: 0;">Upload and ingest data to start asking questions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Render existing conversation with custom avatars
for turn in st.session_state.chat_history:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(turn["answer"])

# Chat input
user_query = st.chat_input("Chat with MyKart Assistant…")

if user_query:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    similar_chunks = st.session_state.db.similarity_search(user_query, k=4)
    context        = "\n\n".join(chunk.page_content.strip() for chunk in similar_chunks)
    messages       = build_prompt(context, st.session_state.chat_history, user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("🛒 Buzzing through the KB…"):
            answer = llm.invoke(messages).content.strip()
        st.markdown(answer)

    st.session_state.chat_history.append({"question": user_query, "answer": answer})