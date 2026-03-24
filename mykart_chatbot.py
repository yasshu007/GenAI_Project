# MyKart Online Store Assistant
import os
import json
import base64
import streamlit as st
import hashlib
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio

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
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="50" fill="#3A3A3A"/>
  <circle cx="50" cy="36" r="16" fill="#F5C518"/>
  <ellipse cx="50" cy="80" rx="26" ry="20" fill="#F5C518"/>
</svg>
"""

# Assistant avatar: stylized bee in MyKart orange
ASSISTANT_AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="50" fill="#E8500A"/>
  <!-- Body -->
  <ellipse cx="50" cy="58" rx="14" ry="18" fill="#3A3A3A"/>
  <!-- Stripes -->
  <rect x="36" y="53" width="28" height="5" rx="2" fill="#F5C518"/>
  <rect x="36" y="62" width="28" height="5" rx="2" fill="#F5C518"/>
  <!-- Head -->
  <circle cx="50" cy="38" r="11" fill="#F5C518"/>
  <!-- Eyes -->
  <circle cx="46" cy="37" r="2.5" fill="#3A3A3A"/>
  <circle cx="54" cy="37" r="2.5" fill="#3A3A3A"/>
  <!-- Antennae -->
  <line x1="46" y1="28" x2="41" y2="20" stroke="#3A3A3A" stroke-width="2" stroke-linecap="round"/>
  <circle cx="41" cy="19" r="2.5" fill="#3A3A3A"/>
  <line x1="54" y1="28" x2="59" y2="20" stroke="#3A3A3A" stroke-width="2" stroke-linecap="round"/>
  <circle cx="59" cy="19" r="2.5" fill="#3A3A3A"/>
  <!-- Wings -->
  <ellipse cx="34" cy="48" rx="11" ry="7" fill="white" fill-opacity="0.75" transform="rotate(-20 34 48)"/>
  <ellipse cx="66" cy="48" rx="11" ry="7" fill="white" fill-opacity="0.75" transform="rotate(20 66 48)"/>
</svg>
"""

def svg_to_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.strip().encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"

USER_AVATAR      = svg_to_data_uri(USER_AVATAR_SVG)
ASSISTANT_AVATAR = svg_to_data_uri(ASSISTANT_AVATAR_SVG)

# ── MyKart Logo SVG (for header) ─────────────────────────────────────────────
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 80">
  <rect width="140" height="80" rx="12" fill="#E8500A"/>
  <ellipse cx="42" cy="46" rx="10" ry="13" fill="#3A3A3A"/>
  <rect x="32" y="41" width="20" height="4" rx="2" fill="#F5C518"/>
  <rect x="32" y="48" width="20" height="4" rx="2" fill="#F5C518"/>
  <circle cx="42" cy="31" r="9" fill="#F5C518"/>
  <circle cx="39" cy="30" r="2" fill="#3A3A3A"/>
  <circle cx="45" cy="30" r="2" fill="#3A3A3A"/>
  <line x1="39" y1="23" x2="35" y2="16" stroke="#3A3A3A" stroke-width="1.8" stroke-linecap="round"/>
  <circle cx="34" cy="15" r="2" fill="#3A3A3A"/>
  <line x1="45" y1="23" x2="49" y2="16" stroke="#3A3A3A" stroke-width="1.8" stroke-linecap="round"/>
  <circle cx="50" cy="15" r="2" fill="#3A3A3A"/>
  <ellipse cx="30" cy="38" rx="9" ry="6" fill="white" fill-opacity="0.8" transform="rotate(-15 30 38)"/>
  <ellipse cx="54" cy="38" rx="9" ry="6" fill="white" fill-opacity="0.8" transform="rotate(15 54 38)"/>
  <text x="68" y="34" font-family="'Trebuchet MS', sans-serif" font-size="22" font-weight="800" fill="white">AI</text>
  <text x="68" y="60" font-family="'Trebuchet MS', sans-serif" font-size="22" font-weight="800" fill="white">Bees</text>
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

    return True, f"✅ **{pdf_file.name}** ingested — {len(chunks)} chunks added to knowledge base."

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
            "Be concise, warm, encouraging, and professional — in the spirit of MyKart Academy. "
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
    page_icon="🐝",
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
        <h1>Online Store<span>Assistant</span></h1>
        <p>🐝 Incremental RAG &nbsp;·&nbsp; Gemini 2.5 Flash &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; MyKart Store</p>
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
    st.caption("Each PDF is merged into the shared knowledge base.")
 
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.button("➕ Ingest into Knowledge Base"):
            with st.spinner("🐝 Embedding and merging…"):
                ingested, message = ingest_pdf(uploaded_file)
            st.markdown(message)
            if ingested:
                st.session_state.db = load_combined_index()
       
    st.markdown("---")
    st.caption("Ingesting Sales data from SQL to knowledge base.")
    if st.button("🗑️ Ingest Sales Data to KB"):
        from db_data_extract.sql_data_exec import extract_and_prepare_data    
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

        st.markdown("✅ Sales data ingested into knowledge base.")

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
        "© 2026 MyKart Academy<br/>All rights reserved</div>",
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
        <div style="font-size: 3rem;">🐝</div>
        <h3 style="color: #E8500A; margin: 10px 0 6px 0;">Welcome to MyKart Online Store Assistant</h3>
        <p style="color: #666; margin: 0;">Upload and ingest a PDF from the sidebar to start asking questions.</p>
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
user_query = st.chat_input("Ask the MyKart Online Store Assistant anything…")

if user_query:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    similar_chunks = st.session_state.db.similarity_search(user_query, k=4)
    context        = "\n\n".join(chunk.page_content.strip() for chunk in similar_chunks)
    messages       = build_prompt(context, st.session_state.chat_history, user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("🐝 Buzzing through the knowledge base…"):
            answer = llm.invoke(messages).content.strip()
        st.markdown(answer)

    st.session_state.chat_history.append({"question": user_query, "answer": answer})