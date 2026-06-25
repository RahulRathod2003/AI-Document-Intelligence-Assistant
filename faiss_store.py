import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Transformers use self-attention mechanism.",
    "BERT is used for NLP tasks.",
    "FAISS helps in similarity search."
]

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

query = "What is self-attention?"

query_embedding = model.encode([query])

distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=1)

print("Best Match:", chunks[indices[0][0]])
print("Distance:", distances)