
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return the Hugging Face embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model


def generate_embeddings(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings
) -> list[list[float]]:
    """
    Convert document chunks into numerical vectors.
    """

    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_model.embed_documents(texts)

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    return embeddings
