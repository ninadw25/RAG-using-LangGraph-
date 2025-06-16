import streamlit as st
from pdf_agent import rag_agent
import os

# Set page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 PDF RAG Chatbot")
st.markdown("Ask questions about the PDF document and get AI-powered answers!")

# Sidebar for information
with st.sidebar:
    st.header("📖 About")
    st.write("This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on the content of a PDF document.")
    
    st.header("⚙️ How it works")
    st.write("""
    1. The PDF is loaded and split into chunks
    2. Text chunks are embedded and stored in a vector database
    3. When you ask a question, relevant chunks are retrieved
    4. An LLM generates an answer based on the retrieved context
    """)
    
    st.header("📄 Current Document")
    st.write("attention.pdf")

# Check if GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY environment variable not found. Please set your Groq API key.")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the PDF..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call the rag_agent function from pdf_agent.py
                response = rag_agent(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"❌ An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit, LangChain, and Groq API"
    "</div>", 
    unsafe_allow_html=True
)