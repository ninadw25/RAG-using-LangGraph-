from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    # all-MiniLM-L6-v2 is ~82 MB and CPU-friendly
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
