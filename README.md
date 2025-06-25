# 📄 Document RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) chatbot powered by a local LLaMA model (Mistral) using the LangChain framework. It enables users to upload documents (PDFs, Word, Excel, images), ask questions in natural language, and get contextual answers by retrieving relevant document chunks and generating accurate responses.

---

## 🚀 Features

- ✅ Upload support: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.png`, `.jpg`, `.jpeg`
- ✅ Document ingestion and storage with **Chroma DB**
- ✅ Embedded using **HuggingFace BGE-large** model
- ✅ RAG-based question answering using local **LLaMA.cpp** (Mistral 7B Instruct)
- ✅ Query correction with **SymSpell**
- ✅ Semantic rephrasing with **Sentence-BERT**
- ✅ Intent classification with **zero-shot classifier**
- ✅ Confidence-based response generation
- ✅ Logs every interaction with full metadata
- ✅ Simple and modern chat UI using **Streamlit**

---

## 📁 File Structure

```
.
├── app.py                           # Main Streamlit app
├── utils.py                         # Helper functions (read/save files to DB)
├── cragbot_feedback_log.json        # Log file for user-bot interactions
├── frequency_dictionary_en_82_765.txt # SymSpell dictionary file
├── Mistral-7B-Instruct-v0.3.Q4_K_M.gguf # LLaMA model file (needs to be downloaded)
├── Data_db/                         # Chroma vector DB (auto-created)
├── requirements.txt                 # Python dependencies
```

---

## 🔧 Setup Instructions

> ⚠️ Recommended Python Version: **3.10**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document-rag-assistant.git
cd document-rag-assistant
```

### 2. Create and Activate Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not provided, you can generate it:

```bash
pip freeze > requirements.txt
```

### 4. Download Required Files

| File                                     | Purpose                  | Location              |
| ---------------------------------------- | ------------------------ | --------------------- |
| `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` | Quantized LLaMA model    | Place in project root |
| `frequency_dictionary_en_82_765.txt`   | SymSpell typo correction | Place in project root |

### 5. Run the App

```bash
./venv/bin/streamlit run app.py
```

Visit `http://localhost:8501` in your browser to interact with the chatbot.

---

## 🧠 How It Works

1. **File Upload**You upload PDFs, Word, Excel, or image files. The app parses and stores text content into Chroma DB.
2. **Query Handling Pipeline**

   - 🔎 **Typo Correction** – via SymSpell
   - 🤖 **Semantic Matching** – BERT-based cosine similarity
   - 🧠 **Intent Detection** – Zero-shot classification
   - 📚 **RAG Retrieval** – Retrieves document chunks via Chroma
   - 💬 **Answer Generation** – Mistral-7B LLaMA model generates response
3. **Conversation Memory**Maintains a conversation history and returns context-aware responses using `ConversationBufferMemory`.
4. **Confidence Estimation**The system combines similarity, intent, and semantic confidence to decide whether the model is confident in its answer.
5. **Logging**
   Every interaction is logged in `cragbot_feedback_log.json`.

---

## ✅ Future Improvements

- Auto-scroll UI to last message (currently limited by Streamlit behavior)
- Add feedback thumbs-up/down next to each bot message
- Display document sources used in answer generation
- Add authentication (e.g., via Streamlit Community Cloud or Auth0)
- Add PDF viewer for uploaded documents

---

## 💻 Tech Stack

- 🐍 Python 3.10
- 🖼️ Streamlit (UI)
- 🧠 LangChain
- 📚 ChromaDB
- 🔎 HuggingFace Transformers
- 🪶 SentenceTransformers (SBERT)
- 🧩 SymSpellPy
- 🐎 LlamaCpp (Mistral GGUF model)

---

## 🙋‍♀️ Author

**Shalini | Jaival | Himanshu**
Postgraduate in AI & Data Science | Powerlifter & Builder of Intelligent Tools
[GitHub](https://github.com/shalinis97)

---

## 📜 License

MIT License – feel free to use, adapt, and share!

---

## 🔗 Useful Resources

- [LangChain Docs](https://python.langchain.com/)
- [Chroma DB](https://www.trychroma.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Mistral Models](https://mistral.ai/news/)
