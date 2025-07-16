from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_faiss_index(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"Loaded FAISS index from {index_path}")
    return index

# Usage
index = load_faiss_index("faiss_index")




# === Module 5: Load FAISS index and perform similarity search ===
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def search_index(index, query, top_k=3):
    """
    Perform similarity search on the FAISS index.
    Prints top_k results with source and snippet.
    """
    results = index.similarity_search(query, k=top_k)
    print(f"\nTop {top_k} results for query: '{query}'\n")
    ragout=""
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown source")
        snippet = doc.page_content[:500].replace("\n", " ")
        ragout=ragout+f"Result #{i}:"
        ragout=ragout+f"Source: {source}"
        ragout=ragout+f"Content snippet: {snippet}"
        print(f"Result #{i}:")
        print(f"Source: {source}")
        print(f"Content snippet: {snippet}")
        print("-" * 80)
    return ragout





