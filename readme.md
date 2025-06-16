## PDF RAG Question Answering

This repository implements a simple Retrieval-Augmented Generation (RAG) system over a PDF document using:

* **LangChain**: for document loading, text splitting, and vector search (Chroma)
* **Groq LLM**: for answer generation via the `ChatGroq` interface
* **LangGraph**: to define a two-step pipeline (retrieve → answer)
* **Streamlit**: to wrap the RAG agent in a web UI

---
![image](https://github.com/user-attachments/assets/49ddada6-ae1e-4dda-b44c-d6cbaea65f50)
---

### 📂 Repository Structure

```bash
.
├── main.py            # Streamlit UI application
├── rag_agent.py       # Core RAG pipeline and rag_agent() function
├── embedding.py       # User-defined get_embeddings() function
├── attention.pdf      # Sample PDF document for RAG (replaceable)
├── chroma_langchain_db/  # Persisted Chroma vector database
└── requirements.txt   # Python dependencies
```

---

### 🛠️ Core Components

#### 1. `rag_agent.py`

* **PDF Loader & Splitter**: uses `PyPDFLoader` + `RecursiveCharacterTextSplitter` to turn the PDF into 1,000‑token chunks with 200‑token overlaps.
* **Vector Store**: builds or loads a Chroma vector store, persisting embeddings to `./chroma_langchain_db`.
* **LangGraph Pipeline**:

  1. **retrieve**: performs similarity search for the top 4 chunks given the user’s question, storing plain text in `state["context"]`.
  2. **answer**: formats a prompt template with context + question, invokes the Groq LLM, and returns the answer.
* **Public Function**: `rag_agent(question: str) -> str` wraps the pipeline to return a simple string.

#### 2. `main.py`

* **Streamlit UI**:

  * Displays which PDF is used.
  * Text input for user questions.
  * "Get Answer" button triggers `rag_agent`.
  * Shows the model’s answer.

#### 3. `embedding.py`

* Must define `get_embeddings()` which returns a function to compute vector embeddings for text chunks.

