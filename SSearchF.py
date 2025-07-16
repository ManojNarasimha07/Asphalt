from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import pathtocode
def load_faiss_index(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"Loaded FAISS index from {index_path}")
    return index

# Usage
index = load_faiss_index("faiss_index")


# === Perform similarity search and reorder by file ===

def search_index(index, query, top_k=3):
    """
    Perform similarity search on the FAISS index, prioritizing results from different files.
    Prints top_k results with source and snippet.
    """
    # Step 1: Perform the similarity search
    results = index.similarity_search(query, k=top_k * 3)  # Fetch more to allow filtering
    print(f"\nTop results for query: '{query}'\n")
    
    # Step 2: Group results by file and prioritize diverse files
    file_groups = {}
    for doc in results:
        source = doc.metadata.get("source", "Unknown source")
        if source not in file_groups:
            file_groups[source] = []
        file_groups[source].append(doc)
    
    # Step 3: Prioritize results from different files
    ordered_results = []
    for source in file_groups:
        ordered_results.extend(file_groups[source][:1])  # Take one result from each file
    
    # Step 4: If we don’t have enough results, fill up with the rest
    remaining_results = []
    for source in file_groups:
        remaining_results.extend(file_groups[source][1:])
    
    # Merge prioritized and remaining results
    ordered_results.extend(remaining_results)
    
    # Ensure we return only top_k results
    ordered_results = ordered_results[:top_k]
    
    # Output results
    ragout = ""
    sourcelist= []
    for i, doc in enumerate(ordered_results, 1):
        source = doc.metadata.get("source", "Unknown source")
        sourcelist.append(source)
        snippet = doc.page_content[:500].replace("\n", " ")
        ragout += f"Result #{i}:\nSource: {source}\nContent snippet: {snippet}\n{'-' * 80}\n"
        print(f"Result #{i}:")
        print(f"Source: {source}")
        print(f"Content snippet: {snippet}")
        print("-" * 80)

    print(f"Source list: {sourcelist}")
    print("Concatenated content from all files:")
    print(len(pathtocode.read_files_and_concatenate(sourcelist)))
    print("Concatenated content DONE!\n\n")
    return ragout


