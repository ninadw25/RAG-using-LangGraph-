from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph, END
from langchain.schema import Document
from langchain_groq import ChatGroq

from typing_extensions import List, TypedDict

from embedding import get_embeddings
import os

def rag_agent(question):

    file_path = "attention.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)


    vector_store = Chroma(
        collection_name="basic_rag",
        embedding_function=get_embeddings(),
        persist_directory="./chroma_langchain_db",
    )
    vector_store.add_documents(documents=docs)

    #initializing llama using groqAPI
    groq_api_key = os.getenv("GROQ_API_KEY")
    model_name = "llama-3.3-70b-versatile"

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0.6,
        max_tokens=8000
    )

    class QueryState(TypedDict):
        context : List[Document]
        prompt : str
        
    class AnswerState(TypedDict):
        context : List[Document]
        prompt : str
        answer : str
        
    def retrieve(state : QueryState) -> AnswerState:
        """Perform similarity search and store the top-4 chunk texts in state["context"]."""
        documents = vector_store.similarity_search(state["prompt"],k=4)   
        state["context"] = [doc.page_content for doc in documents]
        return state
        
    def llm_call(state : AnswerState) -> AnswerState:
        PROMPT_TEMPLATE = """You are a knowledgeable assistant. Use the following context to answer the user's question.

        Context:
        {context}

        Question:
        {question}

        Answer while refering to the context and focus only on the information provided in the context. If the answer is not in the context, say "I don't know based on the provided document."
        """
        joined_context = "\n---\n".join(state["context"])
        full_prompt = PROMPT_TEMPLATE.format(
            context=joined_context,
            question=state["prompt"]
        )
        response = llm.invoke(full_prompt)
        state["answer"] = response.content.strip()
        return state

    builder = StateGraph(AnswerState,input=QueryState)

    builder.add_node("node_1",retrieve)
    builder.add_node("node_2",llm_call)

    builder.add_edge(START,"node_1")
    builder.add_edge("node_1","node_2")
    builder.add_edge("node_2",END)

    graph = builder.compile()


    # --- Invoke the graph on a prompt ---
    # if __name__ == "__main__":
    # question = "What is the main mechanism of attention described in the paper?"
    
    initial_state: QueryState = {"context": [], "prompt": question}
    result = graph.invoke(initial_state)
    
    return result["answer"]
