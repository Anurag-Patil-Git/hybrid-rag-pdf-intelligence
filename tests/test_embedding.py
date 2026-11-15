
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.embedding import (
    get_embedding_model,
    generate_embeddings
)

import numpy as np


# Step 1: Extract PDF
documents = extract_pdf_pages(
    "data/uploads/sample.pdf"
)

# Step 2: Generate chunks
chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

print("Total chunks:", len(chunks))


# Step 3: Load embedding model
print("\nLoading embedding model...")

embedding_model = get_embedding_model()

print("Embedding model loaded successfully!")


# Step 4: Generate embeddings
embeddings = generate_embeddings(
    chunks,
    embedding_model
)

# Convert to NumPy array
embedding_array = np.array(embeddings)


# Step 5: Inspect results
print("\nEmbedding Results")
print("-------------------------")

print("Number of embeddings:", len(embeddings))
print("Embedding shape:", embedding_array.shape)

print("\nFirst embedding:")
print(embedding_array[0][:10])

print("\nVector norm:")
print(np.linalg.norm(embedding_array[0]))

print("\nEmbedding data type:")
print(embedding_array.dtype)
