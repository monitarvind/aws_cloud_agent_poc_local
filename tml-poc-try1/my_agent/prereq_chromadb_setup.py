# --- Chroma Setup ---
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
knowledge_collection = chroma_client.get_or_create_collection(
    name="knowledge", embedding_function=embed_fn
)

def retrieve_chunks(query: str, top_k: int = 3):
    """Retrieve top_k matches from Chroma"""
    res = knowledge_collection.query(query_texts=[query], n_results=top_k)
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({"text": doc, "metadata": meta, "distance": float(dist)})
    return results

def save_to_chroma(query: str, response: str, source: str="Claude"):
    """Store query-response pair into Chroma"""
    knowledge_collection.add(
        documents=[response],
        metadatas=[{"source": source, "query": query}],
        ids=[f"{hash(query+response)}"]
    )

