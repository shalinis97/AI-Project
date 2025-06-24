
import types
import torch

torch.classes.__path__ = types.SimpleNamespace(_path=[])

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


import os
import sys
import streamlit as st
import json
# Langchain modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document


from utils import read_files,save_to_db,get_file_from_db,init_db

init_db()


from symspellpy import SymSpell, Verbosity
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json
from datetime import datetime



st.set_page_config(page_title="Document RAG Assistant ", layout="centered")




# --- Config ---
persist_directory = 'Data_db'
collection_name = 'Data_collection'
embedding_model_name = "BAAI/bge-large-en"

uploaded_files = st.file_uploader(
        "Upload files (DOCX, Excel, PDF, Images)",
        type=["docx", "xlsx", "xls", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

file_data = {}
add_new_read_data = {}
files_to_process = []

if uploaded_files:
        # Load files
         # Check DB first
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name.lower()
            db_content = get_file_from_db(filename)

            if db_content:
                print("\n\n File name present recovering from DB \n\n")
                file_data[filename] = db_content
            else:
                print("\n\n File name not present reading using library \n\n")

                files_to_process.append(uploaded_file)

        if files_to_process:
            add_new_read_data = read_files(files_to_process) 
            file_data.update(add_new_read_data) 

            for key,val in add_new_read_data.items():

                ext = key.split('.')[-1].lower()
                save_to_db(key, ext, str(val))

        
        st.write("Files:")
        st.write(file_data.keys())

# Prepare documents as a list
documents = []

for filename, content in file_data.items():
    documents.append(Document(
        page_content=content,
        metadata={"source": filename,
                  "id": filename}
    ))

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

# Langchain expects a list of dicts or simple strings here
texts = text_splitter.split_documents(documents)

# --- Embedding ---
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# --- Load Chroma DB (or create if missing) ---
chroma_path = os.path.join(persist_directory, 'chroma.sqlite3')

# Case 1: DB exists
if os.path.exists(chroma_path):
    db = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    
    # 🔍 Step 5: Check existing document IDs (via metadata)
    existing_metas = db._collection.get(include=['metadatas'])["metadatas"]
    existing_ids = {meta.get("id") for meta in existing_metas if meta and "id" in meta}

    # ✅ Step 6: Filter only new documents
    new_texts = [text for text in texts if text.metadata.get("id") not in existing_ids]

    if new_texts:
        print("📥 Ingesting new documents into existing DB...")
        db.add_documents(new_texts)
    else:
        print("🔍 No new documents to ingest. Proceeding with retrieval.")
# Case 2: First-time run — create DB with initial texts
else:
    if not texts:
        # Provide dummy if nothing is given (prevents crash)
        texts = [Document(page_content="Placeholder until real data is added.")]
        print("⚠️ No documents found. Using placeholder.")
    
    print("🆕 Creating new Chroma DB...")
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

# --- Setup Retriever ---
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

# Load LlamaCpp model using LangChain’s community wrapper
@st.cache_resource(show_spinner="Loading LLaMA model...")
def load_llm():
    return LlamaCpp(
    model_path="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
    temperature=0.0,
    max_tokens=1024,
    top_p=0.9,
    n_ctx=4096,
    n_threads=12,
    n_gpu_layers=24
)


llm = load_llm()


# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)


def log_interaction(user_query, corrected_query, rephrased_query, bot_response, confidence):
    log_entry = {
        'timestamp': str(datetime.now()),
        'user_query': user_query,
        'corrected_query': corrected_query,
        'rephrased_query': rephrased_query,
        'bot_response': bot_response,
        'confidence': confidence
    }
    with open('cragbot_feedback_log.json', 'a') as log_file:
        json.dump(log_entry, log_file)
        log_file.write('\n')



# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary('frequency_dictionary_en_82_765.txt', term_index=0, count_index=1)

# Load SentenceTransformer for semantic matching (GPU)
semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='mps')


# Known Queries (expand based on your document corpus)
known_queries = [
    "What is the reimbursement policy?",
    "Explain the leave policy.",
    "How to submit expense reports?",
    "What is the finance document approval process?"
]


def calculate_confidence(similarity_score, intent_score, semantic_score):
    return 0.5 * similarity_score + 0.3 * intent_score + 0.2 * semantic_score


def typo_correction(query):
    corrected_words = []
    for word in query.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)


def semantic_correction(query, threshold=0.8):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    known_embeddings = semantic_model.encode(known_queries, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, known_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores)
    best_score = cos_scores[best_match_idx].item()

    if best_score >= threshold:
        return known_queries[best_match_idx], best_score
    else:
        return query, best_score  # use original query if not semantically close enough


intent_classifier = pipeline('zero-shot-classification',
    model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
    device=0)

def detect_intent(query):
    candidate_labels = ['Reimbursement', 'Leave Policy', 'Expense Submission', 'Document Approval', 'General Help']
    result = intent_classifier(query, candidate_labels)
    top_intent = result['labels'][0]
    intent_score = result['scores'][0]
    return top_intent, intent_score



st.markdown("## Document RAG Assistant ")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    query = st.text_input("Chat Input", placeholder="Type your question here...", label_visibility="collapsed")  # ✅ hides it visually
    submitted = st.form_submit_button("Send")

if submitted and query:
    st.session_state.chat_history.append({"role": "user", "message": query})

    corrected_query = typo_correction(query)
    rephrased_query, semantic_score = semantic_correction(corrected_query)
    intent, intent_score = detect_intent(rephrased_query)

    print(f"Corrected Query: {corrected_query}")
    print(f"Rephrased Query: {rephrased_query}")
    print(f"Detected Intent: {intent} (Score: {intent_score})") 
    # Retrieve documents
    docs_with_scores = db.similarity_search_with_score(rephrased_query, k=3)

    if docs_with_scores:
        similarity_score = docs_with_scores[0][1]  # Top document’s score
        docs = [doc[0] for doc in docs_with_scores]
    else:
        similarity_score = 0.0
        docs = []

    if not docs:
        bot_message = "⚠️ No relevant documents found. Try a different query."
    else:
        with st.spinner("🔎 Searching documents..."):
            response = qa_chain.combine_documents_chain.invoke({
                    "input_documents": docs,
                    "question": rephrased_query
                })
         
            final_confidence = calculate_confidence(similarity_score, intent_score, semantic_score)

            if final_confidence < 0.4:
                bot_message = "⚠️ I'm not confident about this answer. Please provide more details."
            else:
                bot_message = response['output_text'] if isinstance(response, dict) else str(response)
            log_interaction(query, corrected_query, rephrased_query, bot_message, final_confidence)

    st.session_state.chat_history.append({"role": "bot", "message": bot_message})


# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {chat['message']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {chat['message']}")