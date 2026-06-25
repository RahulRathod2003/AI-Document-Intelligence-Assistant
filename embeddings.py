from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sample_text = """
Attention mechanism allows models to focus on important information.
Transformers use self-attention.
"""

embedding = model.encode(sample_text)

print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])