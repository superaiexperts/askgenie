# create_vector_store.py

import faiss
import openai
import numpy as np
import pickle
from kb_loader import load_pdf, split_text

openai.api_key = "your-openai-api-key"  # replace with env or manually set

def get_embedding(text, model="text-embedding-ada-002"):
    result = openai.Embedding.create(input=[text], model=model)
    return result['data'][0]['embedding']

def create_vector_store(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

if __name__ == "__main__":
    text = load_pdf("BANK OF PUNE SOP 1.pdf")
    chunks = split_text(text)

    index, chunks = create_vector_store(chunks)

    faiss.write_index(index, "kb.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector Store created successfully!")
